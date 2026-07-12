from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob
from anodyne_dataset.ports import DatasetRepository
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


class _FakeDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self.specs: dict[UUID, DatasetSpec] = {}
        self.jobs: dict[UUID, GenerationJob] = {}
        self.versions: dict[UUID, list[DatasetVersion]] = {}

    async def create_spec(self, spec: DatasetSpec) -> None:
        self.specs[spec.id] = spec

    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        spec = self.specs.get(dataset_id)
        return spec if spec is not None and spec.tenant_id == tenant_id else None

    async def list_specs(self, tenant_id: UUID) -> list[DatasetSpec]:
        return [s for s in self.specs.values() if s.tenant_id == tenant_id]

    async def update_spec(self, spec: DatasetSpec) -> None:
        self.specs[spec.id] = spec

    async def save_job(self, job: GenerationJob) -> None:
        self.jobs[job.id] = job

    async def get_job(self, tenant_id: UUID, job_id: UUID) -> GenerationJob | None:
        job = self.jobs.get(job_id)
        return job if job is not None and job.tenant_id == tenant_id else None

    async def add_version(self, version: DatasetVersion) -> None:
        self.versions.setdefault(version.dataset_id, []).append(version)

    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]:
        return [v for v in self.versions.get(dataset_id, []) if v.tenant_id == tenant_id]


@pytest.fixture
def wired() -> tuple[AsyncClient, Any, _FakeDatasetRepository]:
    app = create_app()
    repo = _FakeDatasetRepository()
    app.dependency_overrides[deps.get_dataset_repo] = lambda: repo
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    return client, app, repo


async def test_list_templates_requires_read(wired):  # type: ignore[no-untyped-def]
    client, app, _ = wired
    r_unauth = await client.get("/templates")
    assert r_unauth.status_code == 401

    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, uuid4())
    r = await client.get("/templates")

    assert r.status_code == 200
    keys = {t["key"] for t in r.json()}
    assert "customers" in keys
    assert "users_churn" in keys


async def test_create_from_template_persists_spec(wired):  # type: ignore[no-untyped-def]
    client, app, repo = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post("/datasets/from-template", json={"template_key": "customers"})

    assert r.status_code == 201
    body = r.json()
    assert body["source"] == "template"
    assert body["tenant_id"] == str(tid)
    assert [f["name"] for f in body["fields"]]  # non-empty, from the template
    assert UUID(body["id"]) in repo.specs


async def test_create_from_template_overrides(wired):  # type: ignore[no-untyped-def]
    client, app, repo = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, uuid4())

    r = await client.post(
        "/datasets/from-template",
        json={
            "template_key": "customers",
            "name": "My customers",
            "target_rows": 42,
            "directives": {
                "directives": [{"kind": "bias", "field": "plan", "value": "pro", "rate": 0.5}]
            },
        },
    )

    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "My customers"
    assert body["target_rows"] == 42
    assert body["directives"]["directives"][0]["field"] == "plan"


async def test_create_from_template_unknown_key_is_404(wired):  # type: ignore[no-untyped-def]
    client, app, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, uuid4())

    r = await client.post("/datasets/from-template", json={"template_key": "does-not-exist"})

    assert r.status_code == 404


async def test_create_from_template_requires_write(wired):  # type: ignore[no-untyped-def]
    client, app, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, uuid4())

    r = await client.post("/datasets/from-template", json={"template_key": "customers"})

    assert r.status_code == 403


async def test_patch_updates_directives(wired):  # type: ignore[no-untyped-def]
    client, app, repo = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post("/datasets/from-template", json={"template_key": "transactions"})
    dataset_id = created.json()["id"]

    r = await client.patch(
        f"/datasets/{dataset_id}",
        json={
            "directives": {
                "directives": [
                    {"kind": "edge_case", "field": "amount", "value": "max", "rate": 0.1}
                ]
            }
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["directives"]["directives"][0]["kind"] == "edge_case"
    assert repo.specs[UUID(dataset_id)].directives["directives"][0]["kind"] == "edge_case"
