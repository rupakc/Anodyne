from __future__ import annotations

from uuid import uuid4

import pytest
from anodyne_evaluation.aggregator import WeightedAggregator
from anodyne_evaluation.models import EvalDimension, EvaluationReport, ExpertScore


def _score(dim: EvalDimension, value: float, recs: list[str] | None = None) -> ExpertScore:
    return ExpertScore(dimension=dim, score=value, rationale="", recommendations=recs or [])


def _agg(scores: list[ExpertScore]) -> EvaluationReport:
    return WeightedAggregator().aggregate(
        scores,
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        reference_version_id=None,
    )


def test_weighted_mean_with_default_weights() -> None:
    # fidelity 0.22 * 1.0 + privacy 0.17 * 0.0 over present weight 0.39.
    report = _agg([_score(EvalDimension.FIDELITY, 1.0), _score(EvalDimension.PRIVACY, 0.0)])
    assert report.overall_score == pytest.approx((0.22 * 1.0 + 0.17 * 0.0) / 0.39)
    # Weights are renormalized over present dimensions and sum to 1.
    assert report.weights["fidelity"] == pytest.approx(0.22 / 0.39)
    assert sum(report.weights.values()) == pytest.approx(1.0)


def test_renormalization_when_experts_missing() -> None:
    # Only fidelity present -> its renormalized weight is 1.0 and overall == its score.
    report = _agg([_score(EvalDimension.FIDELITY, 0.6)])
    assert report.overall_score == pytest.approx(0.6)
    assert report.weights == {"fidelity": pytest.approx(1.0)}


def test_custom_weight_override() -> None:
    report = WeightedAggregator().aggregate(
        [_score(EvalDimension.FIDELITY, 1.0), _score(EvalDimension.BIAS, 0.0)],
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        reference_version_id=None,
        weights={"fidelity": 3.0, "bias": 1.0},
    )
    assert report.overall_score == pytest.approx(0.75)  # (3*1 + 1*0)/4


def test_banding_and_recommendations() -> None:
    strong = _agg([_score(EvalDimension.FIDELITY, 0.95)])
    weak = _agg([_score(EvalDimension.FIDELITY, 0.3, recs=["fix it"])])
    assert "strong" in strong.summary
    assert "needs work" in weak.summary
    assert weak.recommendations == ["fix it"]


def test_no_scores_is_zero() -> None:
    report = _agg([])
    assert report.overall_score == 0.0
    assert report.weights == {}
