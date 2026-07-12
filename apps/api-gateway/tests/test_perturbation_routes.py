"""Route tests for the perturbation API. Offline: Temporal client + repos are
fakes injected via `app.dependency_overrides` (no live server/DB)."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest_asyncio
from anodyne_core.models import Role, TenantContext, User
from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    Modality,
    PerturbationJob,
    SemanticType,
)
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


class _FakeHandle:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(  # type: ignore[no-untyped-def]
        self, workflow, arg, *, id, task_queue, start_signal=None, **kwargs
    ) -> _FakeHandle:
        self.calls.append({"arg": arg, "id": id, "task_queue": task_queue})
        return _FakeHandle(id)


class _FakeRepo:
    """Implements the DatasetRepository + PerturbationRepository surface the
    perturbation routes touch."""

    def __init__(self) -> None:
        self.specs: dict[UUID, DatasetSpec] = {}
        self.versions: dict[UUID, list[DatasetVersion]] = {}
        self.pert_jobs: dict[UUID, PerturbationJob] = {}

    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        s = self.specs.get(dataset_id)
        return s if s and s.tenant_id == tenant_id else None

    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]:
        return [v for v in self.versions.get(dataset_id, []) if v.tenant_id == tenant_id]

    async def save_perturbation_job(self, job: PerturbationJob) -> None:
        self.pert_jobs[job.id] = job

    async def get_perturbation_job(self, tenant_id: UUID, job_id: UUID) -> PerturbationJob | None:
        j = self.pert_jobs.get(job_id)
        return j if j and j.tenant_id == tenant_id else None

    async def list_perturbation_jobs(
        self, tenant_id: UUID, dataset_id: UUID
    ) -> list[PerturbationJob]:
        return [
            j
            for j in self.pert_jobs.values()
            if j.tenant_id == tenant_id and j.dataset_id == dataset_id
        ]


@pytest_asyncio.fixture
async def wired() -> Any:
    app = create_app()
    repo = _FakeRepo()
    tid = uuid4()
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
    repo.specs[spec.id] = spec
    repo.versions[spec.id] = [version]
    client = _FakeTemporalClient()

    app.dependency_overrides[deps.get_dataset_repo] = lambda: repo
    app.dependency_overrides[deps.get_perturbation_repo] = lambda: repo
    app.dependency_overrides[deps.get_temporal_client] = lambda: client
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://t") as ac:
        yield ac, repo, client, tid, spec, version, app
    app.dependency_overrides.clear()


async def test_launch_perturbation_starts_workflow_and_saves_job(wired) -> None:  # type: ignore[no-untyped-def]
    ac, repo, client, tid, spec, version, app = wired
    resp = await ac.post(
        f"/datasets/{spec.id}/versions/{version.id}/perturb",
        json={"family": "noise", "intensity": 0.3, "target_fields": ["age"], "seed": 5},
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["parent_version_id"] == str(version.id)
    assert body["spec"]["family"] == "noise"
    assert body["spec"]["seed"] == 5  # seed persisted on the job spec, not just the workflow input
    assert body["workflow_id"].startswith("pert-")
    # Job persisted + workflow launched on the generation queue.
    assert len(repo.pert_jobs) == 1
    assert next(iter(repo.pert_jobs.values())).spec.seed == 5
    assert client.calls[0]["task_queue"] == "generation"
    assert client.calls[0]["arg"].modality == "tabular"


async def test_launch_perturbation_unknown_version_404(wired) -> None:  # type: ignore[no-untyped-def]
    ac, repo, client, tid, spec, version, app = wired
    resp = await ac.post(
        f"/datasets/{spec.id}/versions/{uuid4()}/perturb", json={"family": "noise"}
    )
    assert resp.status_code == 404


async def test_launch_perturbation_unknown_dataset_404(wired) -> None:  # type: ignore[no-untyped-def]
    ac, repo, client, tid, spec, version, app = wired
    resp = await ac.post(
        f"/datasets/{uuid4()}/versions/{version.id}/perturb", json={"family": "bias"}
    )
    assert resp.status_code == 404


async def test_get_and_list_perturbation_jobs(wired) -> None:  # type: ignore[no-untyped-def]
    ac, repo, client, tid, spec, version, app = wired
    launch = await ac.post(
        f"/datasets/{spec.id}/versions/{version.id}/perturb", json={"family": "drift"}
    )
    job_id = launch.json()["id"]

    got = await ac.get(f"/perturbation-jobs/{job_id}")
    assert got.status_code == 200
    assert got.json()["id"] == job_id

    listed = await ac.get(f"/datasets/{spec.id}/perturbation-jobs")
    assert listed.status_code == 200
    assert [j["id"] for j in listed.json()] == [job_id]

    missing = await ac.get(f"/perturbation-jobs/{uuid4()}")
    assert missing.status_code == 404


async def test_perturbation_requires_write_permission(wired) -> None:  # type: ignore[no-untyped-def]
    ac, repo, client, tid, spec, version, app = wired
    # A viewer may read but not launch.
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, tid)
    resp = await ac.post(
        f"/datasets/{spec.id}/versions/{version.id}/perturb", json={"family": "noise"}
    )
    assert resp.status_code == 403
    listed = await ac.get(f"/datasets/{spec.id}/perturbation-jobs")
    assert listed.status_code == 200
