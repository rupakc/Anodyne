"""`TaskMetricProvider` for `TaskType.SUMMARIZATION`.

Three metrics are an LLM-oracle rubric (`faithfulness`, `coverage`,
`conciseness`) scored as a single batch verdict over the sampled
(document, summary) pairs -- one `complete` call, one integer 1-5 per
criterion, mapped to `/5`. Two metrics are intrinsic and computed on the full
frame: `compression_ratio` (mean length ratio, per-row capped at 1.0 so an
over-long summary never pulls the metric above the "no compression" bound)
and `abstractiveness` (1 minus mean summary/document bigram overlap, i.e. how
much the summary rephrases rather than copies).
"""

from __future__ import annotations

import json
import re

import pandas as pd  # type: ignore[import-untyped]
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.judges.task_metrics.base import (
    TaskMetricError,
    mean_contribution,
    sample_frame,
    strip_json,
)
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, register_provider

_ORACLE_SYSTEM = (
    "You are grading a batch of document/summary pairs for a summarization task. "
    "Considering all pairs below together as a single batch, rate three criteria, each "
    "an INTEGER from 1 (poor) to 5 (excellent): faithfulness (summaries do not "
    "contradict or fabricate facts absent from their document), coverage (summaries "
    "capture the key points of their document), and conciseness (summaries avoid "
    'redundant or unnecessary detail). Return ONLY JSON: {"faithfulness": int, '
    '"coverage": int, "conciseness": int}. No prose outside the JSON.'
)

_RUBRIC_KEYS = ("faithfulness", "coverage", "conciseness")
_TOKEN_RE = re.compile(r"\w+")


def _text(x: object) -> str:
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x)


def _bigrams(text: str) -> set[tuple[str, str]]:
    tokens = _TOKEN_RE.findall(text.lower())
    return {(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)}


def _bigram_overlap(summary: str, document: str) -> float:
    summary_bigrams = _bigrams(summary)
    if len(summary_bigrams) == 0:
        return 0.0
    document_bigrams = _bigrams(document)
    return len(summary_bigrams & document_bigrams) / len(summary_bigrams)


class SummarizationProvider:
    """Standard metrics for the `summarization` task class."""

    task_type = TaskType.SUMMARIZATION

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="faithfulness",
                label="Faithfulness",
                description="LLM-judged 1-5 rating of whether summaries avoid fabrication.",
                requires_llm=True,
            ),
            MetricSpec(
                key="coverage",
                label="Coverage",
                description="LLM-judged 1-5 rating of key-point coverage.",
                requires_llm=True,
            ),
            MetricSpec(
                key="conciseness",
                label="Conciseness",
                description="LLM-judged 1-5 rating of summary concision.",
                requires_llm=True,
            ),
            MetricSpec(
                key="compression_ratio",
                label="Compression ratio",
                description="Mean summary/document character-length ratio (capped at 1.0).",
                requires_llm=False,
            ),
            MetricSpec(
                key="abstractiveness",
                label="Abstractiveness",
                description="1 minus mean summary/document bigram overlap.",
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
        if "document" not in df.columns or "summary" not in df.columns:
            raise TaskMetricError(
                "summarization requires 'document' and 'summary' columns in the subject frame"
            )

        metrics: dict[str, float] = {}
        if "compression_ratio" in selected:
            metrics["compression_ratio"] = self._compression_ratio(df)
        if "abstractiveness" in selected:
            metrics["abstractiveness"] = self._abstractiveness(df)
        if set(_RUBRIC_KEYS) & selected:
            rubric = await self._llm_oracle(ctx, provider, model_config)
            for key in _RUBRIC_KEYS:
                if key in selected:
                    metrics[key] = rubric[key]

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("faithfulness", 1.0) < 0.6:
            recs.append(
                "LLM-judged faithfulness is low; summaries may contradict or fabricate "
                "facts not present in their source document."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=f"Summarization standard metrics for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    @staticmethod
    def _compression_ratio(df: pd.DataFrame) -> float:
        ratios: list[float] = []
        for doc, summ in zip(df["document"], df["summary"], strict=True):
            doc_s = _text(doc)
            if len(doc_s) == 0:
                continue
            ratios.append(min(1.0, len(_text(summ)) / len(doc_s)))
        return sum(ratios) / len(ratios) if ratios else 0.0

    @staticmethod
    def _abstractiveness(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        overlaps = [
            _bigram_overlap(_text(summ), _text(doc))
            for doc, summ in zip(df["document"], df["summary"], strict=True)
        ]
        mean_overlap = sum(overlaps) / len(overlaps)
        return max(0.0, min(1.0, 1.0 - mean_overlap))

    async def _llm_oracle(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
    ) -> dict[str, float]:
        sample = sample_frame(ctx)
        if sample.empty:
            return dict.fromkeys(_RUBRIC_KEYS, 0.0)
        docs = sample["document"].astype(str).tolist()
        summaries = sample["summary"].astype(str).tolist()
        lines = [
            f"{i}. Document: {d}\nSummary: {s}"
            for i, (d, s) in enumerate(zip(docs, summaries, strict=True), start=1)
        ]
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_ORACLE_SYSTEM),
                Message(role="user", content="\n\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields a reproducible verdict.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        return self._parse_rubric(resp.content)

    @staticmethod
    def _parse_rubric(raw: str) -> dict[str, float]:
        text = strip_json(raw)
        try:
            data = json.loads(text)
            out: dict[str, float] = {}
            for key in _RUBRIC_KEYS:
                v = data[key]
                if isinstance(v, bool) or not isinstance(v, int) or not (1 <= v <= 5):
                    raise ValueError(f"{key} must be an integer 1-5, got {v!r}")
                out[key] = v / 5.0
            return out
        except Exception as exc:  # json/validation errors -> domain error
            raise TaskMetricError(
                f"could not parse faithfulness/coverage/conciseness rubric from model output: {exc}"
            ) from exc


register_provider(SummarizationProvider())
