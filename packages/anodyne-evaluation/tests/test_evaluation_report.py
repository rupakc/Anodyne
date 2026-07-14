from __future__ import annotations

import json
from uuid import uuid4

from anodyne_evaluation.models import EvalDimension, EvaluationReport, ExpertScore
from anodyne_evaluation.report import render_html, render_json


def _report() -> EvaluationReport:
    return EvaluationReport(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        reference_version_id=uuid4(),
        overall_score=0.72,
        expert_scores=[
            ExpertScore(
                dimension=EvalDimension.FIDELITY,
                score=0.8,
                rationale="close to reference",
                metrics={"ks_mean": 0.05},
            ),
            ExpertScore(
                dimension=EvalDimension.PRIVACY,
                score=0.6,
                rationale="some proximity",
                metrics={"privacy_risk": 0.4},
            ),
        ],
        weights={"fidelity": 0.56, "privacy": 0.44},
        recommendations=["increase generalization"],
        summary="acceptable overall",
    )


def test_render_json_round_trips() -> None:
    data = render_json(_report())
    parsed = EvaluationReport.model_validate_json(data)
    assert parsed.overall_score == 0.72
    assert json.loads(data)["summary"] == "acceptable overall"


def test_render_html_is_self_contained() -> None:
    html = render_html(_report())
    assert "<!doctype html>" in html.lower()
    assert "72" in html  # overall score rendered as /100
    assert "fidelity" in html.lower()
    assert "increase generalization" in html
    # Fully self-contained: no external network references.
    for token in ("http://", "https://", "src=", "<link", "<script"):
        assert token not in html


def test_render_html_renders_standard_task_metrics() -> None:
    report = _report()
    report.expert_scores.append(
        ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=0.9,
            rationale="binary classification task",
            metrics={"accuracy": 0.9, "macro_f1": 0.88},
        )
    )
    html = render_html(report)
    assert "Standard task metrics" in html
    assert "binary classification task" in html
    assert "accuracy" in html
    assert "macro_f1" in html
    assert "0.900" in html
    assert "0.880" in html
