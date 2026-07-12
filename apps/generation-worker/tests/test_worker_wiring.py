"""Unit tests for `generation_worker.main`'s wiring — no live Temporal/Ray needed.

`temporalio.worker.Worker` is monkeypatched to a recording stub: constructing
a *real* `Worker` requires an already-connected `Client` (verified —
`Client.connect` performs a live gRPC handshake to a Temporal server, and
`Worker.__init__` extracts the client's connected bridge service client;
there is no lazy/offline construction path). Patching `Worker` still
exercises `build_worker`'s actual production code path — which workflows and
activities it registers, on which task queue, and that it binds the injected
fake repo/object-store via `configure_activities` — without needing a live
server.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob
from anodyne_dataset.ports import DatasetRepository
from anodyne_workflows.workflow import GenerationWorkflow
from generation_worker import main
from generation_worker.main import WorkerDeps, build_worker

EXPECTED_ACTIVITY_NAMES = {
    "plan_shards",
    "generate_shards",
    "assemble_and_upload",
    "register_version",
    "set_status",
}


class _FakeObjectStore(ObjectStore):
    async def put(self, key: str, data: bytes) -> None: ...

    async def get(self, key: str) -> bytes:
        return b""

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return []


class _FakeDatasetRepository(DatasetRepository):
    async def create_spec(self, spec: DatasetSpec) -> None: ...

    async def get_spec(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> DatasetSpec | None:
        return None

    async def list_specs(self, tenant_id: uuid.UUID) -> list[DatasetSpec]:
        return []

    async def update_spec(self, spec: DatasetSpec) -> None: ...

    async def save_job(self, job: GenerationJob) -> None: ...

    async def get_job(self, tenant_id: uuid.UUID, job_id: uuid.UUID) -> GenerationJob | None:
        return None

    async def add_version(self, version: DatasetVersion) -> None: ...

    async def list_versions(
        self, tenant_id: uuid.UUID, dataset_id: uuid.UUID
    ) -> list[DatasetVersion]:
        return []


class _FakeWorker:
    """Stand-in for `temporalio.worker.Worker`: just records its constructor args."""

    def __init__(
        self,
        client: object,
        *,
        task_queue: str,
        workflows: list[type],
        activities: list[Callable[..., Any]],
    ) -> None:
        self.client = client
        self.task_queue = task_queue
        self.workflows = workflows
        self.activities = activities


class _FakeClient:
    """A stub in place of a connected `temporalio.client.Client`."""


def test_build_worker_registers_workflow_and_all_five_activities_on_generation_queue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "Worker", _FakeWorker)
    deps = WorkerDeps(repo=_FakeDatasetRepository(), object_store=_FakeObjectStore())

    worker = build_worker(_FakeClient(), deps)  # type: ignore[arg-type]

    assert isinstance(worker, _FakeWorker)
    assert worker.task_queue == "generation"
    assert worker.workflows == [GenerationWorkflow]
    activity_names = {a.__temporal_activity_definition.name for a in worker.activities}
    assert activity_names == EXPECTED_ACTIVITY_NAMES


def test_registered_workflows_and_activities_match_task_queue_constant() -> None:
    assert main.TASK_QUEUE == "generation"
    assert main.registered_workflows() == [GenerationWorkflow]
    names = {a.__temporal_activity_definition.name for a in main.registered_activities()}  # type: ignore[attr-defined]
    assert names == EXPECTED_ACTIVITY_NAMES
