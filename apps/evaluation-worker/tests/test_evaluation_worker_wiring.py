"""Unit tests for `evaluation_worker.main`'s wiring -- no live Temporal/Ray needed.

Mirrors `generation-worker`'s wiring test: `Worker` is monkeypatched to a
recording stub, so `build_worker`'s real path (which workflows/activities it
registers, on which queue, and that it binds the injected fakes via
`configure_evaluation_activities`) is exercised offline.
"""

from __future__ import annotations

import uuid
from typing import Any

import pytest
from anodyne_evaluation.evaluator import sequential_runner
from anodyne_workflows.evaluation_workflow import EvaluationWorkflow
from evaluation_worker import main
from evaluation_worker.main import WorkerDeps, build_worker

EXPECTED_ACTIVITY_NAMES = {"run_evaluation", "set_eval_status"}


class _FakeEvalRepo:
    async def create_run(self, run): ...  # type: ignore[no-untyped-def]
    async def save_run(self, run): ...  # type: ignore[no-untyped-def]
    async def get_run(self, tenant_id, run_id):  # type: ignore[no-untyped-def]
        return None

    async def list_runs(self, tenant_id, dataset_id):  # type: ignore[no-untyped-def]
        return []

    async def add_expert_results(self, tenant_id, run_id, scores): ...  # type: ignore[no-untyped-def]
    async def get_expert_results(self, tenant_id, run_id):  # type: ignore[no-untyped-def]
        return []


class _FakeDatasetRepo:
    async def get_spec(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID):  # type: ignore[no-untyped-def]
        return None

    async def list_versions(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID):  # type: ignore[no-untyped-def]
        return []


class _FakeWorker:
    def __init__(
        self, client: object, *, task_queue: str, workflows: list[type], activities: list[Any]
    ) -> None:
        self.client = client
        self.task_queue = task_queue
        self.workflows = workflows
        self.activities = activities


class _FakeClient: ...


def _deps() -> WorkerDeps:
    return WorkerDeps(
        repo=_FakeEvalRepo(),  # type: ignore[arg-type]
        dataset_repo=_FakeDatasetRepo(),  # type: ignore[arg-type]
        s3_bucket="test-bucket",
        s3_client=None,
        runner=sequential_runner,
    )


def test_build_worker_registers_workflow_and_activities(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(main, "Worker", _FakeWorker)
    worker = build_worker(_FakeClient(), _deps())  # type: ignore[arg-type]
    assert isinstance(worker, _FakeWorker)
    assert worker.task_queue == "evaluation"
    assert worker.workflows == [EvaluationWorkflow]
    names = {a.__temporal_activity_definition.name for a in worker.activities}
    assert names == EXPECTED_ACTIVITY_NAMES


def test_build_worker_binds_context(monkeypatch: pytest.MonkeyPatch) -> None:
    from anodyne_workflows import evaluation_activities as ea

    monkeypatch.setattr(main, "Worker", _FakeWorker)
    deps = _deps()
    build_worker(_FakeClient(), deps)  # type: ignore[arg-type]
    ctx = ea._context()  # noqa: SLF001
    assert ctx.repo is deps.repo
    assert ctx.dataset_repo is deps.dataset_repo
    assert ctx.runner is sequential_runner


def test_task_queue_constant() -> None:
    assert main.TASK_QUEUE == "evaluation"
    assert main.registered_workflows() == [EvaluationWorkflow]
    names = {a.__temporal_activity_definition.name for a in main.registered_activities()}
    assert names == EXPECTED_ACTIVITY_NAMES
