"""`MoEEvaluator`: run the mixture of experts, then aggregate their verdicts.

The judges are run through a pluggable `JudgeRunner` (default `sequential_runner`,
used in tests and non-Ray deployments; the worker injects
`anodyne_compute.ray_evaluation.RayJudgeRunner` for parallel fan-out). Judges
that raise `JudgeNotApplicable` are silently dropped -- the aggregator then
renormalizes over the dimensions that applied.
"""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.aggregator import WeightedAggregator
from anodyne_evaluation.judges import (
    BiasJudge,
    DiversityJudge,
    FidelityJudge,
    PrivacyJudge,
    QualitativeJudge,
    UtilityJudge,
)
from anodyne_evaluation.models import EvaluationReport, ExpertScore
from anodyne_evaluation.ports import (
    Aggregator,
    EvaluationContext,
    Judge,
    JudgeNotApplicable,
    JudgeRunner,
)


async def sequential_runner(judges: Sequence[Judge], ctx: EvaluationContext) -> list[ExpertScore]:
    """Await each judge in turn, dropping the ones that don't apply."""
    out: list[ExpertScore] = []
    for judge in judges:
        try:
            out.append(await judge.evaluate(ctx))
        except JudgeNotApplicable:
            continue
    return out


def default_judges(
    provider: LLMProvider | None = None, model_config: ModelConfig | None = None
) -> list[Judge]:
    """The full mixture of experts. The qualitative (LLM) expert is included
    only when an `LLMProvider` + `ModelConfig` are available."""
    judges: list[Judge] = [
        FidelityJudge(),
        DiversityJudge(),
        PrivacyJudge(),
        UtilityJudge(),
        BiasJudge(),
    ]
    if provider is not None and model_config is not None:
        judges.append(QualitativeJudge(provider, model_config))
    return judges


class MoEEvaluator:
    def __init__(
        self,
        judges: Sequence[Judge],
        aggregator: Aggregator | None = None,
        runner: JudgeRunner = sequential_runner,
    ) -> None:
        self._judges = list(judges)
        self._aggregator = aggregator or WeightedAggregator()
        self._runner = runner

    async def evaluate(
        self,
        ctx: EvaluationContext,
        *,
        tenant_id: UUID,
        dataset_id: UUID,
        dataset_version_id: UUID,
        reference_version_id: UUID | None = None,
        weights: dict[str, float] | None = None,
    ) -> EvaluationReport:
        scores = await self._runner(self._judges, ctx)
        return self._aggregator.aggregate(
            scores,
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            reference_version_id=reference_version_id,
            weights=weights,
        )
