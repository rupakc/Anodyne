"""`TaskMetricProvider` for `TaskType.QA`.

Two metrics are intrinsic (`answerable_rate`, `question_type_diversity`,
computed on the full subject frame) and two are an LLM-oracle
(`answer_correctness`, `groundedness`): a single `complete` call asks the model
to judge every sampled (question, answer) pair -- with a `context`/`document`
column folded in when present -- returning parallel boolean arrays that are
reduced to fractions-true. When neither the frame nor the row supplies a
grounding context, the model is still asked to judge groundedness from the
question/answer pair alone (there is no separate ungrounded-judgment mode --
keeping one prompt/one call simpler than branching the rubric).
"""

from __future__ import annotations

import json
import re

import pandas as pd  # type: ignore[import-untyped]
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.judges.task_metrics.base import (
    TaskMetricError,
    is_nonempty,
    mean_contribution,
    sample_frame,
    strip_json,
)
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, register_provider

_ORACLE_SYSTEM = (
    "You are grading question-answering pairs. For each numbered item below, decide "
    "whether the answer correctly and completely answers the question, and whether the "
    "answer is grounded in (supported by) the provided context -- or, if no context is "
    "given, whether the answer's claims are plausible and internally supported. Return "
    'ONLY JSON: {"correct": [<bool for item 1>, ...], "grounded": [<bool for item 1>, '
    "...]} -- two arrays with exactly one boolean per item, in the same order as the "
    "items below. No prose outside the JSON."
)

_INTERROGATIVES = {"what", "why", "how", "when", "where", "who", "which"}
_WORD_RE = re.compile(r"[a-zA-Z]+")


def _question_type(question: object) -> str:
    m = _WORD_RE.search(str(question).lower())
    if m and m.group(0) in _INTERROGATIVES:
        return m.group(0)
    return "other"


class QAProvider:
    """Standard metrics for the `qa` task class."""

    task_type = TaskType.QA

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="answer_correctness",
                label="Answer correctness",
                description="Fraction of sampled answers an LLM oracle judges as correct.",
                requires_llm=True,
            ),
            MetricSpec(
                key="groundedness",
                label="Groundedness",
                description="Fraction of sampled answers an LLM oracle judges as grounded.",
                requires_llm=True,
            ),
            MetricSpec(
                key="answerable_rate",
                label="Answerable rate",
                description="Fraction of rows with a non-empty answer.",
                requires_llm=False,
            ),
            MetricSpec(
                key="question_type_diversity",
                label="Question type diversity",
                description=(
                    "Normalized count of distinct leading-interrogative question types "
                    "(what/why/how/when/where/who/which/other)."
                ),
                requires_llm=False,
            ),
        ]

    async def score(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        *,
        selected: frozenset[str],
    ) -> ExpertScore:
        df = ctx.subject
        if "question" not in df.columns or "answer" not in df.columns:
            raise TaskMetricError(
                "qa requires 'question' and 'answer' columns in the subject frame"
            )

        metrics: dict[str, float] = {}
        if "answerable_rate" in selected:
            metrics["answerable_rate"] = self._answerable_rate(df)
        if "question_type_diversity" in selected:
            metrics["question_type_diversity"] = self._question_type_diversity(df)
        if "answer_correctness" in selected or "groundedness" in selected:
            correctness, groundedness = await self._llm_oracle(ctx, provider, model_config)
            if "answer_correctness" in selected:
                metrics["answer_correctness"] = correctness
            if "groundedness" in selected:
                metrics["groundedness"] = groundedness

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("answer_correctness", 1.0) < 0.7:
            recs.append(
                "LLM-oracle answer correctness on the sampled Q/A pairs is below 0.7; "
                "review answer generation for accuracy."
            )
        if metrics.get("groundedness", 1.0) < 0.7:
            recs.append(
                "LLM-oracle groundedness is below 0.7; answers may not be well-supported "
                "by their context."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=f"QA standard metrics for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    @staticmethod
    def _answerable_rate(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        return sum(1 for a in df["answer"] if is_nonempty(a)) / len(df)

    @staticmethod
    def _question_type_diversity(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        distinct = {_question_type(q) for q in df["question"]}
        return min(1.0, len(distinct) / 7.0)

    async def _llm_oracle(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
    ) -> tuple[float, float]:
        sample = sample_frame(ctx)
        if sample.empty:
            return 0.0, 0.0
        ctx_col = (
            "context"
            if "context" in sample.columns
            else ("document" if "document" in sample.columns else None)
        )
        questions = sample["question"].astype(str).tolist()
        answers = sample["answer"].astype(str).tolist()
        contexts = sample[ctx_col].astype(str).tolist() if ctx_col else None
        lines: list[str] = []
        for i, (q, a) in enumerate(zip(questions, answers, strict=True), start=1):
            entry = f"{i}. Question: {q}\nAnswer: {a}"
            if contexts is not None:
                entry += f"\nContext: {contexts[i - 1]}"
            lines.append(entry)
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_ORACLE_SYSTEM),
                Message(role="user", content="\n\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields reproducible judgments.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        correct, grounded = self._parse_bools(resp.content, expected=len(questions))
        return sum(correct) / len(correct), sum(grounded) / len(grounded)

    @staticmethod
    def _parse_bools(raw: str, *, expected: int) -> tuple[list[bool], list[bool]]:
        text = strip_json(raw)
        try:
            data = json.loads(text)
            correct = data["correct"]
            grounded = data["grounded"]
            if not isinstance(correct, list) or not isinstance(grounded, list):
                raise ValueError(f"'correct'/'grounded' must be arrays, got {data!r}")
            if len(correct) != expected or len(grounded) != expected:
                raise ValueError(
                    f"expected {expected} entries, got correct={len(correct)}, "
                    f"grounded={len(grounded)}"
                )
            return [bool(x) for x in correct], [bool(x) for x in grounded]
        except Exception as exc:  # json/validation errors -> domain error
            raise TaskMetricError(
                f"could not parse correctness/groundedness judgments from model output: {exc}"
            ) from exc


register_provider(QAProvider())
