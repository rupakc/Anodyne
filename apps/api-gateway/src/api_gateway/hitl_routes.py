"""Human-in-the-Loop & Annotation API (sub-system G): annotate dataset-version
rows, leave feedback on a dataset version or an evaluation run, and drive the
generalized review queue.

A focused `APIRouter` mounted by `create_app` (`build_router()`, matching
`evaluation_routes`). Tenant ownership is enforced the same way as every other
routes module: every read/write resolves its target through the caller's own
tenant-scoped repository (RLS + explicit `tenant_id` filter), so one tenant can
never annotate, give feedback on, or review another tenant's data.

`ReviewTask` generalizes `GenerationWorkflow.approve_schema`'s signal +
`wait_condition` gate (`anodyne_workflows.workflow`): `apply_review_decision`
resolves the Temporal signal to send (falling back to
`anodyne_hitl.models.default_signal_name` per `ReviewKind` when the task
didn't set one explicitly) and applies it -- `approve` signals the paused
workflow to resume, `reject` cancels it (Temporal's built-in, workflow-agnostic
abort -- generalizes to any workflow, not just ones with a bespoke reject
signal), `changes_requested` only persists the decision.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal
from uuid import UUID, uuid4

from anodyne_core.models import TenantContext
from anodyne_dataset.ports import DatasetRepository
from anodyne_evaluation.ports import EvaluationRepository
from anodyne_hitl.models import (
    Annotation,
    Feedback,
    ReviewStatus,
    ReviewTask,
    TargetType,
    Thumbs,
    default_signal_name,
)
from anodyne_hitl.ports import AnnotationRepository, FeedbackRepository, ReviewRepository
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from temporalio.client import Client

from api_gateway import deps

Decision = Literal["approve", "reject", "changes_requested"]

_STATUS_BY_DECISION: dict[Decision, ReviewStatus] = {
    "approve": ReviewStatus.APPROVED,
    "reject": ReviewStatus.REJECTED,
    "changes_requested": ReviewStatus.CHANGES_REQUESTED,
}


class CreateAnnotationRequest(BaseModel):
    row_index: int | None = None
    record_id: str | None = None
    label: str | None = None
    tags: list[str] = Field(default_factory=list)
    comment: str | None = None


class CreateFeedbackRequest(BaseModel):
    target_type: TargetType
    target_id: UUID
    rating: int | None = Field(default=None, ge=1, le=5)
    thumbs: Thumbs | None = None
    comment: str | None = None
    expert_override: dict[str, object] | None = None


class ReviewDecisionRequest(BaseModel):
    decision: Decision
    comment: str | None = None


async def apply_review_decision(
    client: Client,
    review_repo: ReviewRepository,
    task: ReviewTask,
    decision: Decision,
    comment: str | None,
) -> ReviewTask:
    """Resolve + send the Temporal signal/cancel for `task`, then persist the
    decision. Pure enough to be exercised directly against a real
    `GenerationWorkflow` (see `test_hitl_review_gate_integration.py`), not just
    through the route."""
    if task.workflow_id is not None:
        handle = client.get_workflow_handle(task.workflow_id)
        if decision == "approve":
            signal_name = task.signal_name or default_signal_name(task.kind)
            if signal_name is not None:
                await handle.signal(signal_name)
        elif decision == "reject":
            await handle.cancel()
        # changes_requested: no workflow action -- the caller is expected to
        # edit and resubmit, there is nothing to resume or abort yet.
    task.status = _STATUS_BY_DECISION[decision]
    task.reviewer_comment = comment
    task.decided_at = datetime.now(UTC)
    await review_repo.save(task)
    return task


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/datasets/{dataset_id}/versions/{version_id}/annotations", status_code=201)
    async def create_annotation(
        dataset_id: UUID,
        version_id: UUID,
        body: CreateAnnotationRequest,
        ctx: TenantContext = Depends(deps.require("annotations:write")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
        annotation_repo: AnnotationRepository = Depends(deps.get_annotation_repo),
    ) -> dict[str, object]:
        spec = await repo.get_spec(ctx.tenant_id, dataset_id)
        if spec is None:
            raise HTTPException(404, "dataset not found")
        version = await repo.get_version(ctx.tenant_id, version_id)
        if version is None or version.dataset_id != dataset_id:
            raise HTTPException(404, "version not found")
        annotation = Annotation(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            dataset_id=dataset_id,
            version_id=version_id,
            row_index=body.row_index,
            record_id=body.record_id,
            label=body.label,
            tags=body.tags,
            comment=body.comment,
            author=ctx.user.email,
        )
        await annotation_repo.add(annotation)
        return annotation.model_dump(mode="json")

    @router.get("/datasets/{dataset_id}/versions/{version_id}/annotations")
    async def list_annotations(
        dataset_id: UUID,
        version_id: UUID,
        ctx: TenantContext = Depends(deps.require("annotations:read")),
        annotation_repo: AnnotationRepository = Depends(deps.get_annotation_repo),
    ) -> list[dict[str, object]]:
        rows = await annotation_repo.list_for_version(ctx.tenant_id, dataset_id, version_id)
        return [a.model_dump(mode="json") for a in rows]

    @router.delete("/annotations/{annotation_id}", status_code=204)
    async def delete_annotation(
        annotation_id: UUID,
        ctx: TenantContext = Depends(deps.require("annotations:write")),
        annotation_repo: AnnotationRepository = Depends(deps.get_annotation_repo),
    ) -> None:
        deleted = await annotation_repo.delete(ctx.tenant_id, annotation_id)
        if not deleted:
            raise HTTPException(404, "annotation not found")

    @router.post("/feedback", status_code=201)
    async def create_feedback(
        body: CreateFeedbackRequest,
        ctx: TenantContext = Depends(deps.require("annotations:write")),
        dataset_repo: DatasetRepository = Depends(deps.get_dataset_repo),
        eval_repo: EvaluationRepository = Depends(deps.get_evaluation_repo),
        feedback_repo: FeedbackRepository = Depends(deps.get_feedback_repo),
    ) -> dict[str, object]:
        if body.target_type == "dataset_version":
            version = await dataset_repo.get_version(ctx.tenant_id, body.target_id)
            if version is None:
                raise HTTPException(404, "dataset version not found")
        else:
            run = await eval_repo.get_run(ctx.tenant_id, body.target_id)
            if run is None:
                raise HTTPException(404, "evaluation run not found")
        feedback = Feedback(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            target_type=body.target_type,
            target_id=body.target_id,
            rating=body.rating,
            thumbs=body.thumbs,
            comment=body.comment,
            expert_override=body.expert_override,
            author=ctx.user.email,
        )
        await feedback_repo.add(feedback)
        return feedback.model_dump(mode="json")

    @router.get("/reviews")
    async def list_reviews(
        status: ReviewStatus | None = None,
        ctx: TenantContext = Depends(deps.require("reviews:read")),
        review_repo: ReviewRepository = Depends(deps.get_review_repo),
    ) -> list[dict[str, object]]:
        rows = await review_repo.list_for_tenant(ctx.tenant_id, status)
        return [r.model_dump(mode="json") for r in rows]

    @router.get("/reviews/{review_id}")
    async def get_review(
        review_id: UUID,
        ctx: TenantContext = Depends(deps.require("reviews:read")),
        review_repo: ReviewRepository = Depends(deps.get_review_repo),
    ) -> dict[str, object]:
        task = await review_repo.get(ctx.tenant_id, review_id)
        if task is None:
            raise HTTPException(404, "review not found")
        return task.model_dump(mode="json")

    @router.post("/reviews/{review_id}/decision")
    async def decide_review(
        review_id: UUID,
        body: ReviewDecisionRequest,
        ctx: TenantContext = Depends(deps.require("reviews:write")),
        review_repo: ReviewRepository = Depends(deps.get_review_repo),
        client: Client = Depends(deps.get_temporal_client),
    ) -> dict[str, object]:
        task = await review_repo.get(ctx.tenant_id, review_id)
        if task is None:
            raise HTTPException(404, "review not found")
        if task.status != ReviewStatus.PENDING:
            raise HTTPException(409, "review already decided")
        task = await apply_review_decision(client, review_repo, task, body.decision, body.comment)
        return task.model_dump(mode="json")

    return router
