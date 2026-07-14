"""`TaskMetricProvider` for `TaskType.CHAT`.

Two metrics are an LLM-oracle rubric (`instruction_following`, `coherence`)
scored as a single batch verdict over the sampled (instruction, response)
pairs -- one `complete` call, one integer 1-5 per criterion, mapped to `/5`.
`turn_validity` is intrinsic and computed on the full frame: the fraction of
rows where both the instruction and the response are non-empty.
"""

from __future__ import annotations

import json

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
    "You are grading a batch of instruction/response pairs from a chat assistant. "
    "Considering all pairs below together as a single batch, rate two criteria, each an "
    "INTEGER from 1 (poor) to 5 (excellent): instruction_following (responses do what "
    "their instruction asks) and coherence (responses are internally consistent and "
    'well-formed). Return ONLY JSON: {"instruction_following": int, "coherence": int}. '
    "No prose outside the JSON."
)

_RUBRIC_KEYS = ("instruction_following", "coherence")


class ChatProvider:
    """Standard metrics for the `chat` task class."""

    task_type = TaskType.CHAT

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="instruction_following",
                label="Instruction following",
                description="LLM-judged 1-5 rating of whether responses follow their instruction.",
                requires_llm=True,
            ),
            MetricSpec(
                key="coherence",
                label="Coherence",
                description="LLM-judged 1-5 rating of response coherence.",
                requires_llm=True,
            ),
            MetricSpec(
                key="turn_validity",
                label="Turn validity",
                description="Fraction of rows with both a non-empty instruction and response.",
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
        if "instruction" not in df.columns or "response" not in df.columns:
            raise TaskMetricError(
                "chat requires 'instruction' and 'response' columns in the subject frame"
            )

        metrics: dict[str, float] = {}
        if "turn_validity" in selected:
            metrics["turn_validity"] = self._turn_validity(df)
        if set(_RUBRIC_KEYS) & selected:
            rubric = await self._llm_oracle(ctx, provider, model_config)
            for key in _RUBRIC_KEYS:
                if key in selected:
                    metrics[key] = rubric[key]

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("instruction_following", 1.0) < 0.6:
            recs.append(
                "LLM-judged instruction-following is low; responses may not be doing "
                "what their instruction asks."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=f"Chat standard metrics for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    @staticmethod
    def _turn_validity(df: pd.DataFrame) -> float:
        if len(df) == 0:
            return 0.0
        valid = sum(
            1
            for instr, resp in zip(df["instruction"], df["response"], strict=True)
            if is_nonempty(instr) and is_nonempty(resp)
        )
        return valid / len(df)

    async def _llm_oracle(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
    ) -> dict[str, float]:
        sample = sample_frame(ctx)
        if sample.empty:
            return dict.fromkeys(_RUBRIC_KEYS, 0.0)
        instructions = sample["instruction"].astype(str).tolist()
        responses = sample["response"].astype(str).tolist()
        lines = [
            f"{i}. Instruction: {instr}\nResponse: {resp}"
            for i, (instr, resp) in enumerate(zip(instructions, responses, strict=True), start=1)
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
                f"could not parse instruction_following/coherence rubric from model output: {exc}"
            ) from exc


register_provider(ChatProvider())
