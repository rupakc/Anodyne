"""HITL domain models (sub-system G, requirements 12 + 13).

`Annotation`/`Feedback` capture human input on a dataset version (optionally
one row/record) or on an evaluation run. `ReviewTask` generalizes the existing
`GenerationWorkflow.approve_schema` signal + `wait_condition` gate
(`anodyne_workflows.workflow`): any workflow that needs human sign-off creates
one, and approving/rejecting/requesting-changes on it is what resumes/aborts
the paused workflow (see `api_gateway.hitl_routes.apply_review_decision`).
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

TargetType = Literal["dataset_version", "evaluation_run"]
Thumbs = Literal["up", "down"]


class ReviewKind(StrEnum):
    SCHEMA_APPROVAL = "schema_approval"
    DATASET_REVIEW = "dataset_review"
    EVALUATION_REVIEW = "evaluation_review"


class ReviewStatus(StrEnum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    CHANGES_REQUESTED = "changes_requested"


# The one gate that exists today (`GenerationWorkflow.approve_schema`), kept as
# a lookup table rather than hardcoded per call site: a new workflow kind that
# grows a real HITL gate only needs one new entry here.
_DEFAULT_SIGNAL_BY_KIND: dict[ReviewKind, str] = {
    ReviewKind.SCHEMA_APPROVAL: "approve_schema",
}


def default_signal_name(kind: ReviewKind) -> str | None:
    """The Temporal signal name that resumes a paused workflow of this review
    kind, when one hasn't been supplied explicitly on the `ReviewTask`."""
    return _DEFAULT_SIGNAL_BY_KIND.get(kind)


class Annotation(BaseModel):
    """Human label/tag/comment on a dataset version, optionally scoped to one
    row (`row_index`) or record (`record_id`) within it."""

    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    version_id: UUID
    row_index: int | None = None
    record_id: str | None = None
    label: str | None = None
    tags: list[str] = Field(default_factory=list)
    comment: str | None = None
    author: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class Feedback(BaseModel):
    """Rating/thumbs/comment on a dataset version or an evaluation run.

    `expert_override` is a dimension -> value mapping letting a human correct
    a judge's verdict (e.g. `{"fidelity": 0.4}`) without re-running the MoE
    evaluation.
    """

    id: UUID
    tenant_id: UUID
    target_type: TargetType
    target_id: UUID
    rating: int | None = None  # 1-5
    thumbs: Thumbs | None = None
    comment: str | None = None
    expert_override: dict[str, object] | None = None
    author: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ReviewTask(BaseModel):
    """A pending/decided human-review gate on some target (dataset, a
    generation job's schema, an evaluation run, ...). `workflow_id` +
    `signal_name` are set when approving/rejecting should resume/abort a
    paused Temporal workflow (see `default_signal_name`); both are `None` for
    review tasks with no workflow attached (e.g. a plain dataset review)."""

    id: UUID
    tenant_id: UUID
    kind: ReviewKind
    target_type: str
    target_id: UUID
    workflow_id: str | None = None
    signal_name: str | None = None
    status: ReviewStatus = ReviewStatus.PENDING
    reviewer_comment: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    decided_at: datetime | None = None
