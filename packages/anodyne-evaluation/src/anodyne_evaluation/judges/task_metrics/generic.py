"""`TaskMetricProvider` for `TaskType.GENERIC` -- the fallback task class.

Every other provider added in later tasks follows this shape: a fixed rubric
prompt sent through the `LLMProvider` port at `temperature=0`, a fence-stripped
JSON reply, and a `metrics` dict whose caller-selected subset is averaged into
the single `ExpertScore` for `EvalDimension.TASK_QUALITY`. Generic has no
task-specific signal to add, so it reuses the qualitative judge's realism /
coherence / task-fit rubric (see `anodyne_evaluation.judges.qualitative`)
verbatim, mapping each 1-5 score to 0..1.
"""

from __future__ import annotations

import json

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

_RUBRIC = (
    "You are an expert data reviewer judging the quality of a sample of records. "
    "Rate the sample on three criteria, each an INTEGER from 1 (poor) to 5 (excellent): "
    "realism (do values look like plausible real-world data), coherence (are fields "
    "internally consistent with each other), and task_fit (is the data fit for its stated "
    'purpose). Return ONLY JSON: {"realism": int, "coherence": int, "task_fit": int, '
    '"rationale": str}. No prose outside the JSON.'
)


class GenericTaskProvider:
    """Fallback task-quality provider: the shared realism/coherence/task-fit rubric."""

    task_type = TaskType.GENERIC

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="realism",
                label="Realism",
                description="Do sampled values look like plausible real-world data.",
                requires_llm=True,
            ),
            MetricSpec(
                key="coherence",
                label="Coherence",
                description="Are fields internally consistent with each other.",
                requires_llm=True,
            ),
            MetricSpec(
                key="task_fit",
                label="Task fit",
                description="Is the data fit for its stated purpose.",
                requires_llm=True,
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
        sample_text = self._render_sample(ctx)
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_RUBRIC),
                Message(
                    role="user",
                    content=f"Purpose: {ctx.metadata.get('description', 'synthetic dataset')}\n\n"
                    f"Sample records:\n{sample_text}",
                ),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields a reproducible verdict.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        realism, coherence, task_fit, rationale = self._parse(resp.content)
        metrics = {
            "realism": realism / 5.0,
            "coherence": coherence / 5.0,
            "task_fit": task_fit / 5.0,
        }
        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if min(realism, coherence, task_fit) <= 2:
            recs.append(
                "The LLM judge flagged low realism/coherence/task-fit; inspect the "
                "highlighted issues."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=rationale
            or f"Generic task-quality rubric verdict for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    def _render_sample(self, ctx: EvaluationContext) -> str:
        df = sample_frame(ctx)
        if ctx.text_column and ctx.text_column in df.columns:
            docs = df[ctx.text_column].astype(str).tolist()
            return "\n---\n".join(docs)
        lines: list[str] = []
        for i, (_, row) in enumerate(df.iterrows(), start=1):
            fields = ", ".join(f"{c}: {row[c]}" for c in df.columns)
            lines.append(f"{i}. {fields}")
        return "\n".join(lines)

    @staticmethod
    def _parse(raw: str) -> tuple[int, int, int, str]:
        text = strip_json(raw)
        try:
            data = json.loads(text)
            return (
                int(data["realism"]),
                int(data["coherence"]),
                int(data["task_fit"]),
                str(data.get("rationale", "")),
            )
        except Exception as exc:  # json/validation errors -> domain error
            raise TaskMetricError(
                f"could not parse rubric verdict from model output: {exc}"
            ) from exc


register_provider(GenericTaskProvider())
