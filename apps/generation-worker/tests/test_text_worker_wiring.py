"""`WorkerDeps`/`build_worker` gain a model registry + secret key for text
generation. No live Temporal/Ray/DB needed -- mirrors `test_worker_wiring.py`'s
technique (patch `Worker` with a recording stub) but additionally inspects the
`ActivityContext` `build_worker` binds via `configure_activities`.
"""

from __future__ import annotations

import uuid
from typing import Any

import anodyne_workflows.activities as activities_module
import pytest
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob
from anodyne_dataset.ports import DatasetRepository
from generation_worker import main
from generation_worker.main import WorkerDeps, build_worker


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

    async def get_version(
        self, tenant_id: uuid.UUID, version_id: uuid.UUID
    ) -> DatasetVersion | None:
        return None


class _FakeModelRegistry:
    async def get(self, tenant_id: uuid.UUID, config_id: uuid.UUID) -> None:
        return None


class _FakeWorker:
    def __init__(
        self,
        client: object,
        *,
        task_queue: str,
        workflows: list[Any],
        activities: list[Any],
    ) -> None:
        pass


class _FakeClient:
    pass


def test_build_worker_binds_model_registry_and_secret_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "Worker", _FakeWorker)
    registry = _FakeModelRegistry()
    deps = WorkerDeps(
        repo=_FakeDatasetRepository(),
        s3_bucket="test-bucket",
        s3_client=None,
        model_registry=registry,
        secret_key="k",
    )

    build_worker(_FakeClient(), deps)  # type: ignore[arg-type]

    ctx = activities_module._context()
    assert ctx.model_registry is registry
    assert ctx.secret_key == "k"


def test_build_worker_defaults_model_registry_and_secret_key_when_omitted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(main, "Worker", _FakeWorker)
    deps = WorkerDeps(repo=_FakeDatasetRepository(), s3_bucket="test-bucket", s3_client=None)

    build_worker(_FakeClient(), deps)  # type: ignore[arg-type]

    ctx = activities_module._context()
    assert ctx.model_registry is None
    assert ctx.secret_key == ""
