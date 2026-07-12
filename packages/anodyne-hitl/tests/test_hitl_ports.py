"""Ports are ABCs: verify each abstract method name exists and a minimal
in-memory implementation satisfies the contract (mirrors
`anodyne-evaluation`'s port tests -- there are none dedicated, so this
establishes the same "ABC is instantiable once implemented" smoke check used
implicitly by every Sql* adapter in this repo)."""

from __future__ import annotations

from uuid import UUID, uuid4

from anodyne_hitl.models import Annotation, Feedback, ReviewKind, ReviewStatus, ReviewTask
from anodyne_hitl.ports import AnnotationRepository, FeedbackRepository, ReviewRepository


class _InMemoryAnnotations(AnnotationRepository):
    def __init__(self) -> None:
        self.rows: dict[UUID, Annotation] = {}

    async def add(self, annotation: Annotation) -> None:
        self.rows[annotation.id] = annotation

    async def list_for_version(
        self, tenant_id: UUID, dataset_id: UUID, version_id: UUID
    ) -> list[Annotation]:
        return [
            a
            for a in self.rows.values()
            if a.tenant_id == tenant_id
            and a.dataset_id == dataset_id
            and a.version_id == version_id
        ]

    async def delete(self, tenant_id: UUID, annotation_id: UUID) -> bool:
        a = self.rows.get(annotation_id)
        if a is None or a.tenant_id != tenant_id:
            return False
        del self.rows[annotation_id]
        return True


class _InMemoryFeedback(FeedbackRepository):
    def __init__(self) -> None:
        self.rows: list[Feedback] = []

    async def add(self, feedback: Feedback) -> None:
        self.rows.append(feedback)

    async def list_for_target(
        self, tenant_id: UUID, target_type: str, target_id: UUID
    ) -> list[Feedback]:
        return [
            f
            for f in self.rows
            if f.tenant_id == tenant_id
            and f.target_type == target_type
            and f.target_id == target_id
        ]


class _InMemoryReviews(ReviewRepository):
    def __init__(self) -> None:
        self.rows: dict[UUID, ReviewTask] = {}

    async def create(self, task: ReviewTask) -> None:
        self.rows[task.id] = task

    async def get(self, tenant_id: UUID, review_id: UUID) -> ReviewTask | None:
        t = self.rows.get(review_id)
        return t if t and t.tenant_id == tenant_id else None

    async def list_for_tenant(
        self, tenant_id: UUID, status: ReviewStatus | None = None
    ) -> list[ReviewTask]:
        return [
            t
            for t in self.rows.values()
            if t.tenant_id == tenant_id and (status is None or t.status == status)
        ]

    async def save(self, task: ReviewTask) -> None:
        self.rows[task.id] = task


async def test_annotation_repository_contract() -> None:
    repo = _InMemoryAnnotations()
    tid, did, vid = uuid4(), uuid4(), uuid4()
    a = Annotation(id=uuid4(), tenant_id=tid, dataset_id=did, version_id=vid, author="u")
    await repo.add(a)
    assert await repo.list_for_version(tid, did, vid) == [a]
    assert await repo.delete(tid, a.id) is True
    assert await repo.list_for_version(tid, did, vid) == []


async def test_feedback_repository_contract() -> None:
    repo = _InMemoryFeedback()
    tid, target = uuid4(), uuid4()
    f = Feedback(
        id=uuid4(), tenant_id=tid, target_type="dataset_version", target_id=target, author="u"
    )
    await repo.add(f)
    assert await repo.list_for_target(tid, "dataset_version", target) == [f]


async def test_review_repository_contract() -> None:
    repo = _InMemoryReviews()
    tid = uuid4()
    t = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.SCHEMA_APPROVAL,
        target_type="dataset",
        target_id=uuid4(),
    )
    await repo.create(t)
    assert await repo.get(tid, t.id) == t
    assert await repo.list_for_tenant(tid, ReviewStatus.PENDING) == [t]
    t.status = ReviewStatus.APPROVED
    await repo.save(t)
    assert (await repo.get(tid, t.id)).status == ReviewStatus.APPROVED  # type: ignore[union-attr]
