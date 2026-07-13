from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, GenerationJob, Modality, Profile
from anodyne_dataset.ports import DatasetRepository, ProfileRepository, SchemaProposer
from anodyne_generation.proposer import SchemaProposalError
from anodyne_tabular.profiler import PandasSampleProfiler
from anodyne_workflows.workflow import GenerationInput
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


class _FailingSchemaProposer(SchemaProposer):
    async def propose(self, description: str) -> list[Any]:
        raise SchemaProposalError("should never be called for source='sample'")


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
        job = self.jobs.get(job_id)
        return job if job is not None and job.tenant_id == tenant_id else None

    async def add_version(self, version: Any) -> None: ...

    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[Any]:
        return []

    async def get_version(self, tenant_id: UUID, version_id: UUID) -> Any:
        return None


class _FakeProfileRepository(ProfileRepository):
    def __init__(self) -> None:
        self.profiles: dict[UUID, Profile] = {}

    async def save_profile(self, profile: Profile) -> None:
        self.profiles[profile.dataset_id] = profile

    async def get_profile(self, tenant_id: UUID, dataset_id: UUID) -> Profile | None:
        p = self.profiles.get(dataset_id)
        return p if p is not None and p.tenant_id == tenant_id else None


class _FakeObjectStore(ObjectStore):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def get(self, key: str) -> bytes:
        return self.objects[key]

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return [k for k in self.objects if k.startswith(prefix)]


class _FakeHandle:
    def __init__(self, id: str) -> None:
        self.id = id


class _FakeTemporalClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def start_workflow(  # type: ignore[no-untyped-def]
        self, workflow, arg, *, id, task_queue, start_signal=None, **kwargs
    ) -> _FakeHandle:
        self.calls.append({"arg": arg})
        return _FakeHandle(id)


@pytest.fixture
def wired() -> tuple[AsyncClient, Any, _FakeDatasetRepository, _FakeProfileRepository, Any]:
    app = create_app()
    repo = _FakeDatasetRepository()
    profile_repo = _FakeProfileRepository()
    object_store = _FakeObjectStore()
    fake_client = _FakeTemporalClient()
    app.dependency_overrides[deps.get_dataset_repo] = lambda: repo
    app.dependency_overrides[deps.get_profile_repo] = lambda: profile_repo
    app.dependency_overrides[deps.get_object_store] = lambda: object_store
    app.dependency_overrides[deps.get_schema_proposer] = lambda: _FailingSchemaProposer()
    app.dependency_overrides[deps.get_sample_profiler] = lambda: PandasSampleProfiler()
    # The shared generate route resolves the LLM model registry (used only for
    # text datasets); stub it so a tabular from-sample generate doesn't build a
    # real secret-store-backed registry.
    app.dependency_overrides[deps.get_model_registry] = lambda: None
    app.dependency_overrides[deps.get_temporal_client] = lambda: fake_client
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    return client, app, repo, profile_repo, fake_client


async def test_create_sample_dataset_skips_llm_and_has_empty_fields(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(
        "/datasets", json={"name": "customers", "source": "sample", "target_rows": 0}
    )

    assert r.status_code == 201
    body = r.json()
    assert body["source"] == "sample"
    assert body["fields"] == []


async def test_upload_sample_populates_fields_and_row_count(wired):  # type: ignore[no-untyped-def]
    client, app, repo, profile_repo, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post(
        "/datasets", json={"name": "d", "source": "sample", "target_rows": 0}
    )
    dataset_id = created.json()["id"]
    csv_bytes = b"age,plan\n30,gold\n40,silver\n25,gold\n"

    r = await client.post(
        f"/datasets/{dataset_id}/sample",
        files={"file": ("sample.csv", csv_bytes, "text/csv")},
    )

    assert r.status_code == 200
    body = r.json()
    assert {f["name"] for f in body["dataset"]["fields"]} == {"age", "plan"}
    assert body["dataset"]["target_rows"] == 3
    assert body["profile"]["row_count"] == 3
    got = await client.get(f"/datasets/{dataset_id}")
    assert {f["name"] for f in got.json()["fields"]} == {"age", "plan"}
    assert await profile_repo.get_profile(tid, UUID(dataset_id)) is not None


async def test_upload_sample_rejects_non_sample_dataset(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    app.dependency_overrides[deps.get_schema_proposer] = lambda: _FailingSchemaProposer()
    repo.specs[UUID(int=1)] = DatasetSpec(
        id=UUID(int=1),
        tenant_id=tid,
        name="d",
        description="x",
        modality=Modality.TABULAR,
        source="description",
        fields=[],
        target_rows=10,
    )

    r = await client.post(
        f"/datasets/{UUID(int=1)}/sample",
        files={"file": ("sample.csv", b"a\n1\n", "text/csv")},
    )

    assert r.status_code == 400


async def test_upload_sample_oversized_is_413(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post(
        "/datasets", json={"name": "d", "source": "sample", "target_rows": 0}
    )
    dataset_id = created.json()["id"]
    too_big = b"a" * (26 * 1024 * 1024)

    r = await client.post(
        f"/datasets/{dataset_id}/sample",
        files={"file": ("sample.csv", too_big, "text/csv")},
    )

    assert r.status_code == 413


async def test_upload_sample_requires_write_permission(wired):  # type: ignore[no-untyped-def]
    client, app, repo, _, _ = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post(
        "/datasets", json={"name": "d", "source": "sample", "target_rows": 0}
    )
    dataset_id = created.json()["id"]
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, tid)

    r = await client.post(
        f"/datasets/{dataset_id}/sample",
        files={"file": ("sample.csv", b"a\n1\n", "text/csv")},
    )

    assert r.status_code == 403


async def test_generate_threads_synthesizer_directive_into_method(wired):  # type: ignore[no-untyped-def]
    client, app, repo, profile_repo, fake_client = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post(
        "/datasets", json={"name": "d", "source": "sample", "target_rows": 0}
    )
    dataset_id = created.json()["id"]
    await client.post(
        f"/datasets/{dataset_id}/sample",
        files={"file": ("sample.csv", b"age\n1\n2\n3\n", "text/csv")},
    )
    spec = repo.specs[UUID(dataset_id)]
    spec.directives = {"synthesizer": "ctgan"}
    repo.specs[UUID(dataset_id)] = spec

    r = await client.post(f"/datasets/{dataset_id}/generate", json={"seed": 0})

    assert r.status_code == 202
    inp: GenerationInput = fake_client.calls[0]["arg"]
    assert inp.method == "ctgan"
