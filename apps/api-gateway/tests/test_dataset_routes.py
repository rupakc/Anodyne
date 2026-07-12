from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    GenerationJob,
    JobStatus,
    SemanticType,
)
from anodyne_dataset.ports import DatasetRepository, SchemaProposer
from anodyne_workflows.workflow import GenerationInput, GenerationWorkflow
from api_gateway import deps
from api_gateway.app import create_app
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


_PROPOSED_FIELDS = [
    FieldSpec(name="age", semantic_type=SemanticType.INTEGER),
    FieldSpec(name="email", semantic_type=SemanticType.EMAIL),
]


class _FakeSchemaProposer(SchemaProposer):
    async def propose(self, description: str) -> list[FieldSpec]:
        return list(_PROPOSED_FIELDS)


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


class _FakeHandle:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(  # type: ignore[no-untyped-def]
        self, workflow, arg, *, id, task_queue, start_signal=None, **kwargs
    ) -> _FakeHandle:
        self.calls.append(
            {
                "workflow": workflow,
                "arg": arg,
                "id": id,
                "task_queue": task_queue,
                "start_signal": start_signal,
            }
        )
        return _FakeHandle(id)


class _FakeObjectStore(ObjectStore):
    async def put(self, key: str, data: bytes) -> None:
        pass

    async def get(self, key: str) -> bytes:
        return b""

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return []


class _FakeRedis:
    """Minimal `RedisLike`; pubsub with no queued messages (WS auth tests only)."""

    def pubsub(self) -> _FakePubSub:
        return _FakePubSub()


class _FakePubSub:
    async def subscribe(self, *channels: str) -> None:
        pass

    async def get_message(
        self,
        *,
        ignore_subscribe_messages: bool = False,
        timeout: float | None = None,  # noqa: ASYNC109
    ) -> dict[str, Any] | None:
        return None

    async def unsubscribe(self, *channels: str) -> None:
        pass

    async def close(self) -> None:
        pass


@pytest.fixture
def wired() -> tuple[AsyncClient, Any, _FakeDatasetRepository, _FakeTemporalClient]:
    app = create_app()
    repo = _FakeDatasetRepository()
    fake_client = _FakeTemporalClient()
    app.dependency_overrides[deps.get_dataset_repo] = lambda: repo
    app.dependency_overrides[deps.get_schema_proposer] = lambda: _FakeSchemaProposer()
    app.dependency_overrides[deps.get_temporal_client] = lambda: fake_client
    app.dependency_overrides[deps.get_object_store] = lambda: _FakeObjectStore()
    app.dependency_overrides[deps.get_redis] = lambda: _FakeRedis()
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    return client, app, repo, fake_client


async def test_create_dataset_returns_proposed_schema(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(
        "/datasets", json={"name": "customers", "description": "people", "target_rows": 1000}
    )

    assert r.status_code == 201
    body = r.json()
    assert body["name"] == "customers"
    assert body["target_rows"] == 1000
    assert [f["name"] for f in body["fields"]] == ["age", "email"]
    assert body["tenant_id"] == str(tid)
    # secret-free: dataset specs never carry credentials
    assert "secret_ref" not in body and "api_key" not in body
    assert UUID(body["id"]) in repo.specs


async def test_viewer_cannot_create_dataset(wired):  # type: ignore[no-untyped-def]
    client, app, _, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, uuid4())

    r = await client.post("/datasets", json={"name": "d", "description": "x", "target_rows": 10})

    assert r.status_code == 403


async def test_missing_token_is_401(wired):  # type: ignore[no-untyped-def]
    client, _app, _, _ = wired
    r = await client.get("/datasets")
    assert r.status_code == 401


async def test_patch_updates_schema(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post(
        "/datasets", json={"name": "d", "description": "x", "target_rows": 10}
    )
    dataset_id = created.json()["id"]

    r = await client.patch(
        f"/datasets/{dataset_id}",
        json={
            "name": "renamed",
            "target_rows": 500,
            "fields": [{"name": "id", "semantic_type": "integer"}],
        },
    )

    assert r.status_code == 200
    body = r.json()
    assert body["name"] == "renamed"
    assert body["target_rows"] == 500
    assert [f["name"] for f in body["fields"]] == ["id"]
    assert repo.specs[UUID(dataset_id)].name == "renamed"


async def test_generate_starts_workflow_and_requires_write(wired):  # type: ignore[no-untyped-def]
    client, app, repo, fake_client = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post(
        "/datasets", json={"name": "d", "description": "x", "target_rows": 250}
    )
    dataset_id = created.json()["id"]

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 7})

    assert r.status_code == 202
    job = r.json()
    assert job["dataset_id"] == dataset_id
    assert job["workflow_id"] == f"gen-{job['id']}"
    assert UUID(job["id"]) in repo.jobs
    assert repo.jobs[UUID(job["id"])].workflow_id == f"gen-{job['id']}"

    assert len(fake_client.calls) == 1
    call = fake_client.calls[0]
    assert call["workflow"] is GenerationWorkflow.run
    assert call["id"] == f"gen-{job['id']}"
    assert call["task_queue"] == "generation"
    # Auto-approved at start: C0 does schema review before generate is
    # called, so nothing sends `approve_schema` later -- without this the
    # workflow parks at `awaiting_review` forever.
    assert call["start_signal"] == "approve_schema"
    inp = call["arg"]
    assert isinstance(inp, GenerationInput)
    assert inp.job_id == job["id"]
    assert inp.dataset_id == dataset_id
    assert inp.tenant_id == str(tid)
    assert inp.target_rows == 250
    assert inp.seed == 7

    # viewer cannot start generation
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, tid)
    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 1})
    assert r.status_code == 403


async def test_get_job_status(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post(
        "/datasets", json={"name": "d", "description": "x", "target_rows": 10}
    )
    dataset_id = UUID(created.json()["id"])
    started = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 0})
    job_id = started.json()["id"]

    r = await client.get(f"/jobs/{job_id}")

    assert r.status_code == 200
    body = r.json()
    assert body["id"] == job_id
    assert body["status"] == JobStatus.PENDING.value


async def test_get_unknown_job_is_404(wired):  # type: ignore[no-untyped-def]
    client, app, _, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, uuid4())
    r = await client.get(f"/jobs/{uuid4()}")
    assert r.status_code == 404


async def test_list_versions(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    dataset_id = uuid4()
    await repo.add_version(
        DatasetVersion(
            id=uuid4(), tenant_id=tid, dataset_id=dataset_id, artifact_uri="key/artifact.parquet"
        )
    )

    r = await client.get(f"/datasets/{dataset_id}/versions")

    assert r.status_code == 200
    versions = r.json()
    assert len(versions) == 1
    assert versions[0]["artifact_uri"] == "key/artifact.parquet"


async def test_download_version_returns_presigned_url(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    dataset_id = uuid4()
    version_id = uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id, tenant_id=tid, dataset_id=dataset_id, artifact_uri="k/artifact.parquet"
        )
    )

    r = await client.get(f"/datasets/{dataset_id}/versions/{version_id}/download")

    assert r.status_code == 200
    assert r.json()["url"] == "https://example.test/k/artifact.parquet"


def test_ws_stream_requires_auth() -> None:
    app = create_app()
    app.dependency_overrides[deps.get_redis] = lambda: _FakeRedis()
    client = TestClient(app)
    with pytest.raises(Exception):  # noqa: B017 - starlette raises WebSocketDenialResponse
        with client.websocket_connect(f"/jobs/{uuid4()}/stream"):
            pass


def test_ws_stream_rejects_viewer_without_write_needed_read_ok() -> None:
    # datasets:read is granted to VIEWER, so a viewer *can* open the stream;
    # this asserts the RBAC dependency is actually wired (not skipped) by
    # checking a role with no datasets:read at all is rejected outright.
    app = create_app()
    tid = uuid4()

    class _NoRoleUser:
        pass

    ctx = TenantContext(
        tenant_id=tid,
        user=User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[]),
        roles=[],
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_redis] = lambda: _FakeRedis()
    client = TestClient(app)
    with pytest.raises(Exception):  # noqa: B017 - starlette raises WebSocketDenialResponse
        with client.websocket_connect(f"/jobs/{uuid4()}/stream"):
            pass
