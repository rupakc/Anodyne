"""Cheap unit tests for `anodyne_workflows.activities` using fakes (no Temporal, no infra).

`plan_shards` is exercised as a plain async function (it's pure). `set_status`
is exercised via the context-injection mechanism (`configure_activities`) with
fake `DatasetRepository`/`ObjectStore` implementations.
"""

from __future__ import annotations

import uuid

from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob, JobStatus
from anodyne_dataset.ports import DatasetRepository
from anodyne_workflows.activities import (
    ActivityContext,
    configure_activities,
    plan_shards,
    set_status,
)
from anodyne_workflows.workflow import GenerationInput


class _FakeObjectStore(ObjectStore):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes:
        return b""

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return []


class _FakeDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self.jobs: dict[uuid.UUID, GenerationJob] = {}

    async def create_spec(self, spec: DatasetSpec) -> None: ...

    async def get_spec(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> DatasetSpec | None:
        return None

    async def list_specs(self, tenant_id: uuid.UUID) -> list[DatasetSpec]:
        return []

    async def update_spec(self, spec: DatasetSpec) -> None: ...

    async def save_job(self, job: GenerationJob) -> None:
        self.jobs[job.id] = job

    async def get_job(self, tenant_id: uuid.UUID, job_id: uuid.UUID) -> GenerationJob | None:
        return self.jobs.get(job_id)

    async def add_version(self, version: DatasetVersion) -> None: ...

    async def list_versions(
        self, tenant_id: uuid.UUID, dataset_id: uuid.UUID
    ) -> list[DatasetVersion]:
        return []


def _input(job_id: uuid.UUID, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> GenerationInput:
    return GenerationInput(
        job_id=str(job_id),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=10,
        seed=1,
    )


async def test_set_status_preserves_workflow_id_and_message() -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    repo = _FakeDatasetRepository()
    repo.jobs[job_id] = GenerationJob(
        id=job_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        status=JobStatus.PENDING,
        progress=0.0,
        message="created by gateway",
        workflow_id="wf-x",
    )
    configure_activities(ActivityContext(repo=repo, object_store=_FakeObjectStore()))

    await set_status(_input(job_id, tenant_id, dataset_id), "running", 0.5)

    stored = repo.jobs[job_id]
    assert stored.workflow_id == "wf-x"
    assert stored.message == "created by gateway"  # untouched: no message passed
    assert stored.status == JobStatus.RUNNING
    assert stored.progress == 0.5


async def test_set_status_updates_message_when_passed() -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    repo = _FakeDatasetRepository()
    repo.jobs[job_id] = GenerationJob(
        id=job_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        workflow_id="wf-x",
        message="old",
    )
    configure_activities(ActivityContext(repo=repo, object_store=_FakeObjectStore()))

    await set_status(_input(job_id, tenant_id, dataset_id), "failed", 0.4, message="boom")

    stored = repo.jobs[job_id]
    assert stored.workflow_id == "wf-x"
    assert stored.message == "boom"
    assert stored.status == JobStatus.FAILED


async def test_plan_shards_covers_target_rows_in_contiguous_chunks() -> None:
    inp = GenerationInput(
        job_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        tenant_id=str(uuid.uuid4()),
        target_rows=125_000,
        seed=1,
    )

    shards = await plan_shards(inp)

    assert shards[0][0] == 0
    expected_start = 0
    for start, count in shards:
        assert start == expected_start
        assert count > 0
        expected_start += count
    assert expected_start == inp.target_rows
    assert sum(count for _, count in shards) == inp.target_rows
