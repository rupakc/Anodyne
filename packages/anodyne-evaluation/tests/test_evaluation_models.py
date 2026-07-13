from __future__ import annotations

from uuid import uuid4

from anodyne_evaluation.models import (
    DEFAULT_WEIGHTS,
    GRAPH_WEIGHTS,
    TABULAR_WEIGHTS,
    EvalDimension,
    EvaluationReport,
    EvaluationRun,
    EvaluationStatus,
    ExpertScore,
)


def test_default_weights_sum_to_one() -> None:
    # Each modality group sums to 1.0 on its own; the combined map covers every
    # dimension exactly once (a run is single-modality, so the aggregator only
    # ever renormalizes within one group).
    assert abs(sum(TABULAR_WEIGHTS.values()) - 1.0) < 1e-9
    assert abs(sum(GRAPH_WEIGHTS.values()) - 1.0) < 1e-9
    assert set(DEFAULT_WEIGHTS) == set(EvalDimension)


def test_run_defaults() -> None:
    run = EvaluationRun(
        id=uuid4(), tenant_id=uuid4(), dataset_id=uuid4(), dataset_version_id=uuid4()
    )
    assert run.status is EvaluationStatus.PENDING
    assert run.progress == 0.0
    assert run.overall_score is None


def test_report_round_trips_through_json() -> None:
    report = EvaluationReport(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        overall_score=0.73,
        expert_scores=[
            ExpertScore(
                dimension=EvalDimension.FIDELITY,
                score=0.8,
                rationale="ok",
                metrics={"ks_mean": 0.1},
            )
        ],
        weights={"fidelity": 1.0},
        recommendations=["do X"],
        summary="acceptable",
    )
    again = EvaluationReport.model_validate_json(report.model_dump_json())
    assert again.overall_score == 0.73
    assert again.expert_scores[0].dimension is EvalDimension.FIDELITY
    assert again.expert_scores[0].metrics["ks_mean"] == 0.1
