from __future__ import annotations

from uuid import UUID, uuid4

from anodyne_graph.mapping.aligner import build_mapping_review_task, route_to_review
from anodyne_graph.mapping.models import Mapping, MappingRelation, MappingSet
from anodyne_hitl.models import ReviewKind, ReviewStatus, ReviewTask
from anodyne_hitl.ports import ReviewRepository


class _FakeReviewRepo(ReviewRepository):
    def __init__(self) -> None:
        self.created: list[ReviewTask] = []

    async def create(self, task: ReviewTask) -> None:
        self.created.append(task)

    async def get(self, tenant_id: UUID, review_id: UUID) -> ReviewTask | None:
        return next((t for t in self.created if t.id == review_id), None)

    async def list_for_tenant(
        self, tenant_id: UUID, status: ReviewStatus | None = None
    ) -> list[ReviewTask]:
        return list(self.created)

    async def save(self, task: ReviewTask) -> None:  # pragma: no cover - unused here
        pass


def _mapping(needs_review: bool, subj: str) -> Mapping:
    return Mapping(
        subject_id=subj,
        predicate=MappingRelation.CLOSE_MATCH,
        object_id=subj,
        confidence=0.6 if needs_review else 0.95,
        justification="j",
        matcher="lexical",
        needs_review=needs_review,
    )


def _set(*needs_review_subjects: str) -> MappingSet:
    return MappingSet(
        source_ontology_id="s",
        target_ontology_id="t",
        mappings=[_mapping(True, s) for s in needs_review_subjects]
        + [_mapping(False, "Accepted")],
    )


def test_review_task_created_only_when_flagged() -> None:
    tenant = uuid4()
    artifact = uuid4()
    task = build_mapping_review_task(_set("Org"), tenant_id=tenant, artifact_id=artifact)
    assert task is not None
    assert task.kind == ReviewKind.MAPPING_REVIEW
    assert task.target_type == "ontology_mapping_set"
    assert task.target_id == artifact
    assert task.tenant_id == tenant
    assert task.status == ReviewStatus.PENDING


def test_no_review_task_when_nothing_flagged() -> None:
    ms = MappingSet(
        source_ontology_id="s",
        target_ontology_id="t",
        mappings=[_mapping(False, "Accepted")],
    )
    assert build_mapping_review_task(ms, tenant_id=uuid4(), artifact_id=uuid4()) is None


async def test_route_to_review_persists_single_task_via_repo() -> None:
    repo = _FakeReviewRepo()
    tenant = uuid4()
    artifact = uuid4()
    task = await route_to_review(
        _set("Org", "Loc"), repo, tenant_id=tenant, artifact_id=artifact
    )
    assert task is not None
    # One task per mapping-set (the lighter option), not one per flagged mapping.
    assert len(repo.created) == 1
    assert repo.created[0].id == task.id
    assert repo.created[0].kind == ReviewKind.MAPPING_REVIEW


async def test_route_to_review_noop_when_nothing_flagged() -> None:
    repo = _FakeReviewRepo()
    ms = MappingSet(source_ontology_id="s", target_ontology_id="t", mappings=[])
    task = await route_to_review(ms, repo, tenant_id=uuid4(), artifact_id=uuid4())
    assert task is None
    assert repo.created == []
