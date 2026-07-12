"""Wiring test for the perturbation half of `generation_worker.main`.

Same approach as `test_worker_wiring.py`: `Worker` is monkeypatched to a
recording stub so `build_worker`'s real registration path runs without a live
Temporal server.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from typing import Any

import pytest
from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    GenerationJob,
    PerturbationJob,
    Profile,
)
from anodyne_dataset.ports import DatasetRepository, PerturbationRepository, ProfileRepository
from anodyne_perturbation import RegistryPerturbator
from anodyne_workflows.perturbation_workflow import PerturbationWorkflow
from generation_worker import main
from generation_worker.main import WorkerDeps, build_worker

EXPECTED_PERTURBATION_ACTIVITIES = {
    "set_perturbation_status",
    "apply_perturbation",
    "register_perturbed_version",
}


class _FakeRepo(DatasetRepository, ProfileRepository, PerturbationRepository):
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

    async def save_profile(self, profile: Profile) -> None: ...
    async def get_profile(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> Profile | None:
        return None

    async def save_perturbation_job(self, job: PerturbationJob) -> None: ...
    async def get_perturbation_job(
        self, tenant_id: uuid.UUID, job_id: uuid.UUID
    ) -> PerturbationJob | None:
        return None

    async def list_perturbation_jobs(
        self, tenant_id: uuid.UUID, dataset_id: uuid.UUID
    ) -> list[PerturbationJob]:
        return []


class _FakeWorker:
    def __init__(
        self,
        client: object,
        *,
        task_queue: str,
        workflows: list[type],
        activities: list[Callable[..., Any]],
    ) -> None:
        self.task_queue = task_queue
        self.workflows = workflows
        self.activities = activities


class _FakeClient: ...


def test_registered_perturbation_workflow_and_activities() -> None:
    assert main.registered_perturbation_workflows() == [PerturbationWorkflow]
    names = {
        a.__temporal_activity_definition.name  # type: ignore[attr-defined]
        for a in main.registered_perturbation_activities()
    }
    assert names == EXPECTED_PERTURBATION_ACTIVITIES


def test_build_worker_registers_and_configures_perturbation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from anodyne_workflows import perturbation_activities as pa_mod

    monkeypatch.setattr(main, "Worker", _FakeWorker)
    repo = _FakeRepo()
    deps = WorkerDeps(
        repo=repo,
        s3_bucket="b",
        s3_client=None,
        perturbation_repo=repo,
        perturbator=RegistryPerturbator(),
    )

    worker = build_worker(_FakeClient(), deps)  # type: ignore[arg-type]

    assert isinstance(worker, _FakeWorker)
    assert PerturbationWorkflow in worker.workflows
    names = {a.__temporal_activity_definition.name for a in worker.activities}  # type: ignore[attr-defined]
    assert EXPECTED_PERTURBATION_ACTIVITIES <= names
    ctx = pa_mod._context()  # noqa: SLF001
    assert ctx.perturbation_repo is repo
    assert isinstance(ctx.perturbator, RegistryPerturbator)


def test_perturbation_activities_not_configured_without_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Generation-only deps (no perturbation_repo) still register the perturbation
    # workflow/activities, but leave them unconfigured -- exactly as before D.
    monkeypatch.setattr(main, "Worker", _FakeWorker)
    deps = WorkerDeps(repo=_FakeRepo(), s3_bucket="b", s3_client=None)
    worker = build_worker(_FakeClient(), deps)  # type: ignore[arg-type]
    assert isinstance(worker, _FakeWorker)
    assert PerturbationWorkflow in worker.workflows
