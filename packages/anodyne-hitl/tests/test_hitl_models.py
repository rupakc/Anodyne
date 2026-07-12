from __future__ import annotations

from uuid import uuid4

from anodyne_hitl.models import (
    Annotation,
    Feedback,
    ReviewKind,
    ReviewStatus,
    ReviewTask,
    default_signal_name,
)


def test_annotation_defaults() -> None:
    a = Annotation(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        version_id=uuid4(),
        author="u@x.io",
    )
    assert a.row_index is None
    assert a.record_id is None
    assert a.label is None
    assert a.tags == []
    assert a.comment is None
    assert a.created_at is not None


def test_annotation_with_row_target() -> None:
    a = Annotation(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        version_id=uuid4(),
        row_index=3,
        label="anomaly",
        tags=["pii", "review"],
        comment="looks off",
        author="u@x.io",
    )
    assert a.row_index == 3
    assert a.tags == ["pii", "review"]


def test_feedback_dataset_version_target() -> None:
    f = Feedback(
        id=uuid4(),
        tenant_id=uuid4(),
        target_type="dataset_version",
        target_id=uuid4(),
        rating=4,
        author="u@x.io",
    )
    assert f.rating == 4
    assert f.thumbs is None
    assert f.expert_override is None


def test_feedback_evaluation_run_target_with_expert_override() -> None:
    f = Feedback(
        id=uuid4(),
        tenant_id=uuid4(),
        target_type="evaluation_run",
        target_id=uuid4(),
        thumbs="down",
        expert_override={"fidelity": 0.4},
        author="u@x.io",
    )
    assert f.thumbs == "down"
    assert f.expert_override == {"fidelity": 0.4}


def test_review_task_defaults_pending() -> None:
    t = ReviewTask(
        id=uuid4(),
        tenant_id=uuid4(),
        kind=ReviewKind.SCHEMA_APPROVAL,
        target_type="dataset",
        target_id=uuid4(),
    )
    assert t.status == ReviewStatus.PENDING
    assert t.workflow_id is None
    assert t.signal_name is None
    assert t.decided_at is None


def test_default_signal_name_maps_schema_approval() -> None:
    assert default_signal_name(ReviewKind.SCHEMA_APPROVAL) == "approve_schema"


def test_default_signal_name_none_for_kinds_without_a_gate_yet() -> None:
    assert default_signal_name(ReviewKind.DATASET_REVIEW) is None
    assert default_signal_name(ReviewKind.EVALUATION_REVIEW) is None
