"""Offline unit tests for the perturbation activities, with in-memory fakes for
the repo + object store (no Temporal server, no Ray, no network)."""

from __future__ import annotations

import io
import uuid

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import (
    DatasetVersion,
    JobStatus,
    PerturbationFamily,
    PerturbationJob,
    PerturbationSpec,
)
from anodyne_perturbation import RegistryPerturbator
from anodyne_workflows.perturbation_activities import (
    PerturbationActivityContext,
    apply_perturbation,
    configure_perturbation_activities,
    register_perturbed_version,
    set_perturbation_status,
)
from anodyne_workflows.perturbation_workflow import PerturbationInput


class _FakeStore:
    def __init__(self) -> None:
        self.data: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.data[key] = data

    async def get(self, key: str) -> bytes:
        return self.data[key]

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://x/{key}"

    async def list(self, prefix: str) -> list[str]:
        return [k for k in self.data if k.startswith(prefix)]


class _FakeRepo:
    def __init__(self) -> None:
        self.versions: dict[uuid.UUID, list[DatasetVersion]] = {}
        self.jobs: dict[uuid.UUID, PerturbationJob] = {}

    async def add_version(self, version: DatasetVersion) -> None:
        self.versions.setdefault(version.dataset_id, []).append(version)

    async def list_versions(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID):  # type: ignore[no-untyped-def]
        return [v for v in self.versions.get(dataset_id, []) if v.tenant_id == tenant_id]

    async def save_perturbation_job(self, job: PerturbationJob) -> None:
        self.jobs[job.id] = job

    async def get_perturbation_job(self, tenant_id: uuid.UUID, job_id: uuid.UUID):  # type: ignore[no-untyped-def]
        j = self.jobs.get(job_id)
        return j if j and j.tenant_id == tenant_id else None

    async def list_perturbation_jobs(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return [
            j for j in self.jobs.values() if j.tenant_id == tenant_id and j.dataset_id == dataset_id
        ]


def _parquet_bytes(table: pa.Table) -> bytes:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    return buf.getvalue()


@pytest.fixture
def wired() -> tuple[_FakeRepo, _FakeStore, PerturbationInput, DatasetVersion]:
    tenant, dataset = uuid.uuid4(), uuid.uuid4()
    store = _FakeStore()
    repo = _FakeRepo()
    table = pa.table({"x": pa.array([1.0, 2.0, 3.0, 4.0, 5.0], type=pa.float64())})
    parent = DatasetVersion(
        id=uuid.uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        artifact_uri="datasets/d/gen/artifact.parquet",
        format="parquet",
        row_count=5,
    )
    store.data[parent.artifact_uri] = _parquet_bytes(table)
    repo.versions[dataset] = [parent]
    job = PerturbationJob(
        id=uuid.uuid4(),
        tenant_id=tenant,
        dataset_id=dataset,
        parent_version_id=parent.id,
        spec=PerturbationSpec(family=PerturbationFamily.NOISE, intensity=0.5),
    )
    repo.jobs[job.id] = job
    configure_perturbation_activities(
        PerturbationActivityContext(
            repo=repo,  # type: ignore[arg-type]
            perturbation_repo=repo,  # type: ignore[arg-type]
            perturbator=RegistryPerturbator(),
            s3_bucket="b",
            s3_client=None,
        )
    )
    inp = PerturbationInput(
        job_id=str(job.id),
        dataset_id=str(dataset),
        tenant_id=str(tenant),
        parent_version_id=str(parent.id),
        family="noise",
        intensity=0.5,
        seed=7,
    )
    return repo, store, inp, parent


async def test_apply_perturbation_writes_derived_artifact(monkeypatch, wired) -> None:  # type: ignore[no-untyped-def]
    repo, store, inp, parent = wired
    import anodyne_workflows.perturbation_activities as mod

    monkeypatch.setattr(mod, "_object_store", lambda _inp: store)

    uri, rows = await apply_perturbation(inp)
    assert rows == 5
    assert uri in store.data
    # Derived data differs from the parent artifact (noise applied).
    out = pq.read_table(io.BytesIO(store.data[uri]))
    orig = pq.read_table(io.BytesIO(store.data[parent.artifact_uri]))
    assert out.column("x").to_pylist() != orig.column("x").to_pylist()


async def test_register_perturbed_version_sets_lineage_and_result(monkeypatch, wired) -> None:  # type: ignore[no-untyped-def]
    repo, store, inp, parent = wired
    import anodyne_workflows.perturbation_activities as mod

    monkeypatch.setattr(mod, "_object_store", lambda _inp: store)

    await register_perturbed_version(inp, "datasets/d/perturbations/j/artifact.parquet", 5)
    child = [v for v in repo.versions[parent.dataset_id] if v.id != parent.id][0]
    assert child.parent_version_id == parent.id
    job = repo.jobs[uuid.UUID(inp.job_id)]
    assert job.result_version_id == child.id


async def test_set_perturbation_status_updates_job(wired) -> None:  # type: ignore[no-untyped-def]
    repo, store, inp, parent = wired
    await set_perturbation_status(inp, "running", 0.1, "go")
    job = repo.jobs[uuid.UUID(inp.job_id)]
    assert job.status is JobStatus.RUNNING
    assert job.progress == 0.1
    assert job.message == "go"


async def test_apply_perturbation_rejects_unsupported_format(monkeypatch, wired) -> None:  # type: ignore[no-untyped-def]
    repo, store, inp, parent = wired
    parent.format = "image_manifest"
    import anodyne_workflows.perturbation_activities as mod

    monkeypatch.setattr(mod, "_object_store", lambda _inp: store)
    store.data[parent.artifact_uri] = b"{}"
    with pytest.raises(ValueError, match="parquet/jsonl"):
        await apply_perturbation(inp)
