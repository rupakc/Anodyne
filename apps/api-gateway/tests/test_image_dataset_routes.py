from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from anodyne_core.models import ModelConfig, Role, TenantContext, User
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob, Modality
from anodyne_dataset.ports import DatasetRepository
from anodyne_workflows.workflow import GenerationInput
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


class _FakeImageProviderRegistry:
    def __init__(self) -> None:
        self.configs: dict[UUID, ModelConfig] = {}

    async def create(
        self,
        tenant_id: UUID,
        *,
        name: str,
        provider: str,
        model: str,
        api_key: str | None,
        api_base: str | None,
        params: dict[str, object],
    ) -> ModelConfig:
        cfg = ModelConfig(
            id=uuid4(),
            tenant_id=tenant_id,
            name=name,
            provider=provider,
            model=model,
            params=params,
            secret_ref=f"enc:{api_key}" if api_key else None,
            api_base=api_base,
        )
        self.configs[cfg.id] = cfg
        return cfg

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None:
        cfg = self.configs.get(config_id)
        return cfg if cfg is not None and cfg.tenant_id == tenant_id else None

    async def list(self, tenant_id: UUID) -> list[ModelConfig]:
        return [c for c in self.configs.values() if c.tenant_id == tenant_id]

    async def delete(self, tenant_id: UUID, config_id: UUID) -> None:
        self.configs.pop(config_id, None)


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


@pytest.fixture
def wired() -> tuple[AsyncClient, Any, _FakeDatasetRepository, _FakeImageProviderRegistry]:
    app = create_app()
    repo = _FakeDatasetRepository()
    image_registry = _FakeImageProviderRegistry()
    fake_client = _FakeTemporalClient()
    app.dependency_overrides[deps.get_dataset_repo] = lambda: repo
    app.dependency_overrides[deps.get_image_provider_registry] = lambda: image_registry
    app.dependency_overrides[deps.get_temporal_client] = lambda: fake_client
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    return client, app, repo, image_registry, fake_client  # type: ignore[return-value]


async def test_create_image_dataset_with_labels(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(
        "/datasets/image",
        json={
            "name": "product-photos",
            "description": "a widget on a white background",
            "target_count": 20,
            "labels": ["red", "blue"],
            "directives": {"style": "studio lighting"},
        },
    )

    assert r.status_code == 201
    body = r.json()
    assert body["modality"] == "image"
    assert body["target_rows"] == 20
    assert body["fields"][0]["constraints"]["choices"] == ["red", "blue"]
    assert body["directives"]["style"] == "studio lighting"
    assert UUID(body["id"]) in repo.specs


async def test_create_image_dataset_without_labels_has_no_fields(wired):  # type: ignore[no-untyped-def]
    client, app, _, _, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, uuid4())

    r = await client.post(
        "/datasets/image",
        json={"name": "d", "description": "a cat", "target_count": 5},
    )

    assert r.status_code == 201
    assert r.json()["fields"] == []


async def test_viewer_cannot_create_image_dataset(wired):  # type: ignore[no-untyped-def]
    client, app, _, _, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, uuid4())

    r = await client.post(
        "/datasets/image", json={"name": "d", "description": "x", "target_count": 5}
    )

    assert r.status_code == 403


async def test_generate_on_labelless_image_dataset_does_not_400(wired):  # type: ignore[no-untyped-def]
    """Regression guard: the tabular "no fields" guard must not fire for a
    single-class (no label) image dataset -- only `create_image_dataset`'s
    own validation (none needed here) applies.
    """
    client, app, _, _, fake_client = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post(
        "/datasets/image", json={"name": "d", "description": "a cat", "target_count": 5}
    )
    dataset_id = created.json()["id"]

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 0})

    assert r.status_code == 202
    assert len(fake_client.calls) == 1
    inp = fake_client.calls[0]["arg"]
    assert isinstance(inp, GenerationInput)
    assert inp.modality == Modality.IMAGE.value


async def test_generate_passes_modality_for_image_dataset(wired):  # type: ignore[no-untyped-def]
    client, app, _, _, fake_client = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post(
        "/datasets/image",
        json={"name": "d", "description": "x", "target_count": 5, "labels": ["a"]},
    )
    dataset_id = created.json()["id"]

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 3})

    assert r.status_code == 202
    inp = fake_client.calls[0]["arg"]
    assert inp.modality == "image"
    assert inp.seed == 3


async def test_register_list_and_delete_image_provider(wired):  # type: ignore[no-untyped-def]
    client, app, _, _, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    created = await client.post(
        "/image-providers",
        json={
            "name": "my-openai",
            "provider": "openai-images",
            "model": "dall-e-3",
            "api_key": "sk-secret",
        },
    )
    assert created.status_code == 201
    body = created.json()
    assert "secret_ref" not in body
    config_id = body["id"]

    listed = await client.get("/image-providers")
    assert listed.status_code == 200
    assert len(listed.json()) == 1
    assert "secret_ref" not in listed.json()[0]

    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.ADMIN, tid)
    deleted = await client.delete(f"/image-providers/{config_id}")
    assert deleted.status_code == 204

    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    listed_after = await client.get("/image-providers")
    assert listed_after.json() == []


async def test_viewer_cannot_register_image_provider_but_can_list(wired):  # type: ignore[no-untyped-def]
    client, app, _, _, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, uuid4())

    write = await client.post("/image-providers", json={"name": "n", "provider": "p", "model": "m"})
    assert write.status_code == 403

    read = await client.get("/image-providers")
    assert read.status_code == 200


async def test_member_cannot_delete_image_provider(wired):  # type: ignore[no-untyped-def]
    client, app, _, _, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, uuid4())

    r = await client.delete(f"/image-providers/{uuid4()}")

    assert r.status_code == 403
