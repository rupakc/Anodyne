"""SQL-backed `AnnotationRepository`/`FeedbackRepository`/`ReviewRepository`.

Lives in the `anodyne-hitl` package (not `anodyne-storage`), matching how
`SqlEvaluationRepository` lives in `anodyne-evaluation` while using the shared
`anodyne_storage.db` tables. Every method runs inside a `tenant_session`
(RLS `app.tenant_id` GUC via `SET LOCAL`), and reads add an explicit
`tenant_id` filter as defense-in-depth on top of RLS.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from anodyne_storage.db import dataset_annotations, feedback, review_tasks, tenant_session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from anodyne_hitl.models import Annotation, Feedback, ReviewKind, ReviewStatus, ReviewTask
from anodyne_hitl.ports import AnnotationRepository, FeedbackRepository, ReviewRepository


def _annotation_from_row(m: Any) -> Annotation:
    return Annotation(
        id=m["id"],
        tenant_id=m["tenant_id"],
        dataset_id=m["dataset_id"],
        version_id=m["version_id"],
        row_index=m["row_index"],
        record_id=m["record_id"],
        label=m["label"],
        tags=m["tags"],
        comment=m["comment"],
        author=m["author"],
        created_at=m["created_at"],
    )


def _feedback_from_row(m: Any) -> Feedback:
    return Feedback(
        id=m["id"],
        tenant_id=m["tenant_id"],
        target_type=m["target_type"],
        target_id=m["target_id"],
        rating=m["rating"],
        thumbs=m["thumbs"],
        comment=m["comment"],
        expert_override=m["expert_override"],
        author=m["author"],
        created_at=m["created_at"],
    )


def _review_task_from_row(m: Any) -> ReviewTask:
    return ReviewTask(
        id=m["id"],
        tenant_id=m["tenant_id"],
        kind=ReviewKind(m["kind"]),
        target_type=m["target_type"],
        target_id=m["target_id"],
        workflow_id=m["workflow_id"],
        signal_name=m["signal_name"],
        status=ReviewStatus(m["status"]),
        reviewer_comment=m["reviewer_comment"],
        created_at=m["created_at"],
        decided_at=m["decided_at"],
    )


class SqlAnnotationRepository(AnnotationRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, annotation: Annotation) -> None:
        async with tenant_session(self._engine, annotation.tenant_id) as s:
            await s.execute(
                dataset_annotations.insert().values(
                    id=annotation.id,
                    tenant_id=annotation.tenant_id,
                    dataset_id=annotation.dataset_id,
                    version_id=annotation.version_id,
                    row_index=annotation.row_index,
                    record_id=annotation.record_id,
                    label=annotation.label,
                    tags=annotation.tags,
                    comment=annotation.comment,
                    author=annotation.author,
                    created_at=annotation.created_at,
                )
            )
            await s.commit()

    async def list_for_version(
        self, tenant_id: UUID, dataset_id: UUID, version_id: UUID
    ) -> list[Annotation]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                (
                    await s.execute(
                        select(dataset_annotations).where(
                            dataset_annotations.c.dataset_id == dataset_id,
                            dataset_annotations.c.version_id == version_id,
                            dataset_annotations.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            return [_annotation_from_row(r) for r in rows]

    async def delete(self, tenant_id: UUID, annotation_id: UUID) -> bool:
        # Check-then-delete (rather than trusting `CursorResult.rowcount`,
        # which mypy's SQLAlchemy stubs don't expose on the async `Result`)
        # so this stays type-clean without an `# type: ignore`.
        async with tenant_session(self._engine, tenant_id) as s:
            row = (
                await s.execute(
                    select(dataset_annotations.c.id).where(
                        dataset_annotations.c.id == annotation_id,
                        dataset_annotations.c.tenant_id == tenant_id,
                    )
                )
            ).first()
            if row is None:
                return False
            await s.execute(
                dataset_annotations.delete().where(
                    dataset_annotations.c.id == annotation_id,
                    dataset_annotations.c.tenant_id == tenant_id,
                )
            )
            await s.commit()
            return True


class SqlFeedbackRepository(FeedbackRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add(self, fb: Feedback) -> None:
        async with tenant_session(self._engine, fb.tenant_id) as s:
            await s.execute(
                feedback.insert().values(
                    id=fb.id,
                    tenant_id=fb.tenant_id,
                    target_type=fb.target_type,
                    target_id=fb.target_id,
                    rating=fb.rating,
                    thumbs=fb.thumbs,
                    comment=fb.comment,
                    expert_override=fb.expert_override,
                    author=fb.author,
                    created_at=fb.created_at,
                )
            )
            await s.commit()

    async def list_for_target(
        self, tenant_id: UUID, target_type: str, target_id: UUID
    ) -> list[Feedback]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                (
                    await s.execute(
                        select(feedback).where(
                            feedback.c.target_type == target_type,
                            feedback.c.target_id == target_id,
                            feedback.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            return [_feedback_from_row(r) for r in rows]


class SqlReviewRepository(ReviewRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create(self, task: ReviewTask) -> None:
        async with tenant_session(self._engine, task.tenant_id) as s:
            await s.execute(
                review_tasks.insert().values(
                    id=task.id,
                    tenant_id=task.tenant_id,
                    kind=str(task.kind),
                    target_type=task.target_type,
                    target_id=task.target_id,
                    workflow_id=task.workflow_id,
                    signal_name=task.signal_name,
                    status=str(task.status),
                    reviewer_comment=task.reviewer_comment,
                    created_at=task.created_at,
                    decided_at=task.decided_at,
                )
            )
            await s.commit()

    async def get(self, tenant_id: UUID, review_id: UUID) -> ReviewTask | None:
        async with tenant_session(self._engine, tenant_id) as s:
            row = (
                (
                    await s.execute(
                        select(review_tasks).where(
                            review_tasks.c.id == review_id,
                            review_tasks.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .first()
            )
            return _review_task_from_row(row) if row else None

    async def list_for_tenant(
        self, tenant_id: UUID, status: ReviewStatus | None = None
    ) -> list[ReviewTask]:
        async with tenant_session(self._engine, tenant_id) as s:
            conditions = [review_tasks.c.tenant_id == tenant_id]
            if status is not None:
                conditions.append(review_tasks.c.status == str(status))
            rows = (await s.execute(select(review_tasks).where(*conditions))).mappings().all()
            return [_review_task_from_row(r) for r in rows]

    async def save(self, task: ReviewTask) -> None:
        async with tenant_session(self._engine, task.tenant_id) as s:
            await s.execute(
                review_tasks.update()
                .where(
                    review_tasks.c.id == task.id,
                    review_tasks.c.tenant_id == task.tenant_id,
                )
                .values(
                    status=str(task.status),
                    reviewer_comment=task.reviewer_comment,
                    decided_at=task.decided_at,
                )
            )
            await s.commit()
