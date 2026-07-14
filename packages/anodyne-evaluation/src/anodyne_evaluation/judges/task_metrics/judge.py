"""`TaskMetricsJudge` -- the dispatch judge for the TASK_QUALITY dimension.

Routes to the `TaskMetricProvider` registered for `ctx.task_type` (sub-system
F's per-task standard metrics), narrowing to the caller-selected metric subset
when one was requested. Any precondition failure (no task type resolved, no
provider registered, no valid metrics selected, or the provider itself
failing to parse the model's output) surfaces as `JudgeNotApplicable` so the
aggregator simply excludes this dimension and renormalizes the rest.
"""

from __future__ import annotations

from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge, JudgeNotApplicable
from anodyne_evaluation.task_metrics import catalog_for, provider_for


class TaskMetricsJudge(Judge):
    dimension = EvalDimension.TASK_QUALITY

    def __init__(self, provider: LLMProvider, model_config: ModelConfig) -> None:
        self._provider = provider
        self._cfg = model_config

    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
        task = ctx.task_type
        if task is None:
            raise JudgeNotApplicable("no task type resolved for this run")
        prov = provider_for(task)
        if prov is None:
            raise JudgeNotApplicable(f"no standard-metric provider for task {task}")
        keys = {m.key for m in catalog_for(task)}
        selected = (
            frozenset(ctx.selected_metrics & keys) if ctx.selected_metrics else frozenset(keys)
        )
        if not selected:
            raise JudgeNotApplicable("no valid standard metrics selected")
        try:
            return await prov.score(ctx, self._provider, self._cfg, selected=selected)
        except TaskMetricError as exc:
            raise JudgeNotApplicable(f"task metrics unavailable: {exc}") from exc
