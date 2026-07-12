"""Route tests for the HITL/annotation API (sub-system G). Offline: repos +
Temporal client are fakes injected via `app.dependency_overrides` (no live
server/DB), mirroring `test_perturbation_routes.py`/`test_evaluation_routes.py`."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest_asyncio
from anodyne_core.models import Role, TenantContext, User
from anodyne_dataset.models import DatasetSpec, DatasetVersion, FieldSpec, Modality, SemanticType
from anodyne_evaluation.models import EvaluationRun
from anodyne_hitl.models import Annotation, Feedback, ReviewKind, ReviewStatus, ReviewTask
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="reviewer@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


class _FakeDatasetRepo:
    def __init__(self) -> None:
        self.specs: dict[UUID, DatasetSpec] = {}
        self.versions: dict[UUID, DatasetVersion] = {}

    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        s = self.specs.get(dataset_id)
        return s if s and s.tenant_id == tenant_id else None

    async def get_version(self, tenant_id: UUID, version_id: UUID) -> DatasetVersion | None:
        v = self.versions.get(version_id)
        return v if v and v.tenant_id == tenant_id else None


class _FakeEvalRepo:
    def __init__(self) -> None:
        self.runs: dict[UUID, EvaluationRun] = {}

    async def get_run(self, tenant_id: UUID, run_id: UUID) -> EvaluationRun | None:
        r = self.runs.get(run_id)
        return r if r and r.tenant_id == tenant_id else None


class _FakeAnnotationRepo:
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


class _FakeFeedbackRepo:
    def __init__(self) -> None:
        self.rows: list[Feedback] = []

    async def add(self, feedback: Feedback) -> None:
        self.rows.append(feedback)


class _FakeReviewRepo:
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


class _FakeHandle:
    def __init__(self, workflow_id: str, calls: list[tuple[str, str]]) -> None:
        self._id = workflow_id
        self._calls = calls

    async def signal(self, name: str, *args: Any) -> None:
        self._calls.append(("signal", name))

    async def cancel(self) -> None:
        self._calls.append(("cancel", self._id))


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def get_workflow_handle(self, workflow_id: str) -> _FakeHandle:
        return _FakeHandle(workflow_id, self.calls)


@pytest_asyncio.fixture
async def wired() -> Any:
    app = create_app()
    tid = uuid4()
    dataset_repo = _FakeDatasetRepo()
    eval_repo = _FakeEvalRepo()
    annotation_repo = _FakeAnnotationRepo()
    feedback_repo = _FakeFeedbackRepo()
    review_repo = _FakeReviewRepo()
    client = _FakeTemporalClient()

    spec = DatasetSpec(
        id=uuid4(),
        tenant_id=tid,
        name="d",
        description="x",
        modality=Modality.TABULAR,
        source="description",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=10,
    )
    version = DatasetVersion(id=uuid4(), tenant_id=tid, dataset_id=spec.id, artifact_uri="k")
    dataset_repo.specs[spec.id] = spec
    dataset_repo.versions[version.id] = version

    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    app.dependency_overrides[deps.get_dataset_repo] = lambda: dataset_repo
    app.dependency_overrides[deps.get_evaluation_repo] = lambda: eval_repo
    app.dependency_overrides[deps.get_annotation_repo] = lambda: annotation_repo
    app.dependency_overrides[deps.get_feedback_repo] = lambda: feedback_repo
    app.dependency_overrides[deps.get_review_repo] = lambda: review_repo
    app.dependency_overrides[deps.get_temporal_client] = lambda: client

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        yield (
            ac,
            app,
            tid,
            spec,
            version,
            eval_repo,
            annotation_repo,
            feedback_repo,
            review_repo,
            client,
        )
    app.dependency_overrides.clear()


# --- annotations -------------------------------------------------------------


async def test_create_and_list_annotation(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    resp = await ac.post(
        f"/datasets/{spec.id}/versions/{version.id}/annotations",
        json={"row_index": 3, "label": "anomaly", "tags": ["pii"], "comment": "looks off"},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["row_index"] == 3
    assert body["tags"] == ["pii"]
    assert body["author"] == "reviewer@x.io"

    listed = await ac.get(f"/datasets/{spec.id}/versions/{version.id}/annotations")
    assert listed.status_code == 200
    assert [a["id"] for a in listed.json()] == [body["id"]]


async def test_create_annotation_unknown_dataset_404(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, *_ = wired
    resp = await ac.post(
        f"/datasets/{uuid4()}/versions/{version.id}/annotations", json={"comment": "x"}
    )
    assert resp.status_code == 404


async def test_create_annotation_unknown_version_404(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, *_ = wired
    resp = await ac.post(
        f"/datasets/{spec.id}/versions/{uuid4()}/annotations", json={"comment": "x"}
    )
    assert resp.status_code == 404


async def test_delete_annotation(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, *_ = wired
    created = await ac.post(
        f"/datasets/{spec.id}/versions/{version.id}/annotations", json={"comment": "x"}
    )
    annotation_id = created.json()["id"]
    deleted = await ac.delete(f"/annotations/{annotation_id}")
    assert deleted.status_code == 204
    missing = await ac.delete(f"/annotations/{annotation_id}")
    assert missing.status_code == 404


async def test_annotations_require_write_permission_to_create(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, *_ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, tid)
    resp = await ac.post(
        f"/datasets/{spec.id}/versions/{version.id}/annotations", json={"comment": "x"}
    )
    assert resp.status_code == 403
    listed = await ac.get(f"/datasets/{spec.id}/versions/{version.id}/annotations")
    assert listed.status_code == 200


# --- feedback ----------------------------------------------------------------


async def test_create_feedback_on_dataset_version(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, *_ = wired
    resp = await ac.post(
        "/feedback",
        json={"target_type": "dataset_version", "target_id": str(version.id), "rating": 4},
    )
    assert resp.status_code == 201
    assert resp.json()["rating"] == 4


async def test_create_feedback_on_evaluation_run_with_expert_override(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, *_ = wired
    run = EvaluationRun(
        id=uuid4(), tenant_id=tid, dataset_id=spec.id, dataset_version_id=version.id
    )
    eval_repo.runs[run.id] = run
    resp = await ac.post(
        "/feedback",
        json={
            "target_type": "evaluation_run",
            "target_id": str(run.id),
            "thumbs": "down",
            "expert_override": {"fidelity": 0.4},
        },
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["thumbs"] == "down"
    assert body["expert_override"] == {"fidelity": 0.4}


async def test_create_feedback_unknown_target_404(wired) -> None:  # type: ignore[no-untyped-def]
    ac, *_ = wired
    resp = await ac.post(
        "/feedback", json={"target_type": "dataset_version", "target_id": str(uuid4())}
    )
    assert resp.status_code == 404


# --- reviews -------------------------------------------------------------


async def test_list_reviews_filters_by_status(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    pending = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.SCHEMA_APPROVAL,
        target_type="dataset",
        target_id=spec.id,
        workflow_id="gen-1",
        signal_name="approve_schema",
    )
    approved = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.DATASET_REVIEW,
        target_type="dataset",
        target_id=spec.id,
        status=ReviewStatus.APPROVED,
    )
    await review_repo.create(pending)
    await review_repo.create(approved)

    listed = await ac.get("/reviews", params={"status": "pending"})
    assert listed.status_code == 200
    assert [r["id"] for r in listed.json()] == [str(pending.id)]

    got = await ac.get(f"/reviews/{pending.id}")
    assert got.status_code == 200
    assert got.json()["kind"] == "schema_approval"

    missing = await ac.get(f"/reviews/{uuid4()}")
    assert missing.status_code == 404


async def test_decision_approve_signals_the_paused_workflow(wired) -> None:  # type: ignore[no-untyped-def]
    """Generalizes `GenerationWorkflow.approve_schema`: approving a
    `schema_approval` review task with a `workflow_id` sends that signal."""
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    task = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.SCHEMA_APPROVAL,
        target_type="dataset",
        target_id=spec.id,
        workflow_id="gen-42",
    )
    await review_repo.create(task)

    resp = await ac.post(f"/reviews/{task.id}/decision", json={"decision": "approve"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "approved"
    assert client.calls == [("signal", "approve_schema")]  # defaulted from the kind
    assert review_repo.rows[task.id].status == ReviewStatus.APPROVED
    assert review_repo.rows[task.id].decided_at is not None


async def test_decision_reject_cancels_the_workflow(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    task = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.SCHEMA_APPROVAL,
        target_type="dataset",
        target_id=spec.id,
        workflow_id="gen-42",
    )
    await review_repo.create(task)

    resp = await ac.post(
        f"/reviews/{task.id}/decision", json={"decision": "reject", "comment": "bad schema"}
    )
    assert resp.status_code == 200
    assert resp.json()["status"] == "rejected"
    assert resp.json()["reviewer_comment"] == "bad schema"
    assert client.calls == [("cancel", "gen-42")]


async def test_decision_changes_requested_sends_no_signal(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    task = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.DATASET_REVIEW,
        target_type="dataset",
        target_id=spec.id,
    )
    await review_repo.create(task)

    resp = await ac.post(f"/reviews/{task.id}/decision", json={"decision": "changes_requested"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "changes_requested"
    assert client.calls == []  # no workflow_id -> nothing to signal/cancel


async def test_decision_without_workflow_id_just_persists_status(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    task = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.DATASET_REVIEW,
        target_type="dataset",
        target_id=spec.id,
    )
    await review_repo.create(task)

    resp = await ac.post(f"/reviews/{task.id}/decision", json={"decision": "approve"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "approved"
    assert client.calls == []


async def test_decision_already_decided_is_conflict(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    task = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.DATASET_REVIEW,
        target_type="dataset",
        target_id=spec.id,
        status=ReviewStatus.APPROVED,
    )
    await review_repo.create(task)
    resp = await ac.post(f"/reviews/{task.id}/decision", json={"decision": "reject"})
    assert resp.status_code == 409


async def test_reviews_require_write_permission_to_decide(wired) -> None:  # type: ignore[no-untyped-def]
    ac, app, tid, spec, version, eval_repo, annotation_repo, feedback_repo, review_repo, client = (
        wired
    )
    task = ReviewTask(
        id=uuid4(),
        tenant_id=tid,
        kind=ReviewKind.DATASET_REVIEW,
        target_type="dataset",
        target_id=spec.id,
    )
    await review_repo.create(task)
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, tid)
    resp = await ac.post(f"/reviews/{task.id}/decision", json={"decision": "approve"})
    assert resp.status_code == 403
    listed = await ac.get("/reviews")
    assert listed.status_code == 200
