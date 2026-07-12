"""Weighted 360-degree aggregation of expert verdicts.

overall = sum(w_d * score_d for present d) / sum(w_d for present d)

i.e. a weighted mean over exactly the dimensions that produced a score. Experts
that raised `JudgeNotApplicable` never reach here, so their weight drops out of
both numerator and denominator (renormalization). Weights come from
`DEFAULT_WEIGHTS`, overlaid with any per-run overrides.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from anodyne_evaluation.models import (
    DEFAULT_WEIGHTS,
    EvaluationReport,
    ExpertScore,
)
from anodyne_evaluation.ports import Aggregator


def _band_summary(overall: float, n: int) -> str:
    if overall >= 0.8:
        band = "strong"
    elif overall >= 0.6:
        band = "acceptable"
    else:
        band = "needs work"
    return f"Overall 360-degree quality is {band} ({overall:.2f}) across {n} expert dimension(s)."


class WeightedAggregator(Aggregator):
    def aggregate(
        self,
        scores: list[ExpertScore],
        *,
        tenant_id: UUID,
        dataset_id: UUID,
        dataset_version_id: UUID,
        reference_version_id: UUID | None,
        weights: dict[str, float] | None = None,
    ) -> EvaluationReport:
        effective = {**DEFAULT_WEIGHTS, **(weights or {})}
        present = {s.dimension: float(effective.get(s.dimension, 0.0)) for s in scores}
        total = sum(present.values())

        if not scores:
            overall = 0.0
            applied: dict[str, float] = {}
        elif total <= 0.0:
            # All present dimensions had zero/negative weight: fall back to an
            # unweighted mean so a misconfigured weight map still yields a score.
            overall = sum(s.score for s in scores) / len(scores)
            applied = {str(s.dimension): 1.0 / len(scores) for s in scores}
        else:
            overall = sum(s.score * present[s.dimension] for s in scores) / total
            applied = {str(d): w / total for d, w in present.items()}

        recommendations = [r for s in scores for r in s.recommendations]
        return EvaluationReport(
            id=uuid4(),
            tenant_id=tenant_id,
            dataset_id=dataset_id,
            dataset_version_id=dataset_version_id,
            reference_version_id=reference_version_id,
            overall_score=round(overall, 6),
            expert_scores=scores,
            weights=applied,
            recommendations=recommendations,
            summary=_band_summary(overall, len(scores)),
        )
