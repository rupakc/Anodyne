"""Tests for `POST /datasets/audio`.

Self-contained (rather than importing `test_dataset_routes`'s fixtures):
pytest's `--import-mode=importlib` doesn't add sibling test modules to
`sys.path`, so cross-test-file imports of fixtures are fragile here. The fake
repo/Temporal-client/tenant-context helpers below intentionally mirror
`test_dataset_routes.py`'s shapes.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

from anodyne_core.models import Role, TenantContext, User
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob
from anodyne_dataset.ports import DatasetRepository
from api_gateway import deps
from api_gateway.app import create_app
from fastapi.testclient import TestClient
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


class _FakeDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self.specs: dict[UUID, DatasetSpec] = {}
        self.jobs: dict[UUID, GenerationJob] = {}

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
        return self.jobs.get(job_id)

    async def add_version(self, version: DatasetVersion) -> None: ...

    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]:
        return []


class _FakeHandle:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(  # type: ignore[no-untyped-def]
        self, workflow, arg, *, id, task_queue, start_signal=None, **kwargs
    ) -> _FakeHandle:
        self.calls.append({"workflow": workflow, "arg": arg, "id": id, "task_queue": task_queue})
        return _FakeHandle(id)


def _client() -> tuple[AsyncClient, Any, _FakeDatasetRepository, _FakeTemporalClient]:
    app = create_app()
    repo = _FakeDatasetRepository()
    fake_client = _FakeTemporalClient()
    app.dependency_overrides[deps.get_dataset_repo] = lambda: repo
    app.dependency_overrides[deps.get_temporal_client] = lambda: fake_client
    # The shared generate route resolves the LLM model registry (used only for
    # text datasets); stub it so an audio generate doesn't build a real
    # secret-store-backed registry.
    app.dependency_overrides[deps.get_model_registry] = lambda: None
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    return client, app, repo, fake_client


async def test_create_audio_dataset_returns_audio_modality() -> None:
    client, app, repo, _ = _client()
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(
        "/datasets/audio",
        json={
            "name": "greetings",
            "description": "TTS greetings",
            "target_rows": 3,
            "directives": {"prompts": ["hi", "hello", "hey"], "voice": "narrator"},
        },
    )

    assert r.status_code == 201
    body = r.json()
    assert body["modality"] == "audio"
    assert body["fields"][0]["name"] == "transcript"
    assert body["directives"]["audio"]["prompts"] == ["hi", "hello", "hey"]
    assert body["directives"]["audio"]["voice"] == "narrator"
    assert body["tenant_id"] == str(tid)
    assert UUID(body["id"]) in repo.specs


async def test_create_audio_dataset_defaults_to_empty_directives() -> None:
    client, app, _, _ = _client()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, uuid4())

    r = await client.post("/datasets/audio", json={"name": "d", "target_rows": 5})

    assert r.status_code == 201
    assert r.json()["directives"] == {"audio": {}}


async def test_viewer_cannot_create_audio_dataset() -> None:
    client, app, _, _ = _client()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, uuid4())

    r = await client.post("/datasets/audio", json={"name": "d", "target_rows": 1})

    assert r.status_code == 403


async def test_audio_dataset_can_then_generate() -> None:
    # Proves the existing, unmodified /generate route already works for
    # modality=audio -- no gateway change was needed there.
    client, app, repo, fake_client = _client()
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post("/datasets/audio", json={"name": "d", "target_rows": 2})
    dataset_id = created.json()["id"]

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 1})

    assert r.status_code == 202
    assert len(fake_client.calls) == 1


def test_missing_token_is_401() -> None:
    app = create_app()
    client = TestClient(app)

    r = client.post("/datasets/audio", json={"name": "d", "target_rows": 1})

    assert r.status_code == 401
