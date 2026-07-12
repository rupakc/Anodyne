"""Perturbation workflow orchestration test (mocked activities).

Marked integration: needs the Temporal test-server download, like
`test_workflow.py`. Verifies the workflow dispatches the perturbation activities
by name, in order, and returns the derived artifact uri.
"""

import uuid

import pytest
from anodyne_workflows.perturbation_workflow import PerturbationInput, PerturbationWorkflow
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

pytestmark = pytest.mark.integration


async def test_perturbation_workflow_runs_activity_sequence() -> None:
    calls: list[str] = []

    @activity.defn(name="set_perturbation_status")
    async def set_perturbation_status(inp: PerturbationInput, status: str, progress: float) -> None:
        calls.append(f"status:{status}")

    @activity.defn(name="apply_perturbation")
    async def apply_perturbation(inp: PerturbationInput) -> list[object]:
        calls.append("apply")
        return ["datasets/d/perturbations/j/artifact.parquet", 10]

    @activity.defn(name="register_perturbed_version")
    async def register_perturbed_version(inp: PerturbationInput, uri: str, rows: int) -> None:
        calls.append("register")

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="pert",
            workflows=[PerturbationWorkflow],
            activities=[set_perturbation_status, apply_perturbation, register_perturbed_version],
        ):
            inp = PerturbationInput(
                job_id=str(uuid.uuid4()),
                dataset_id=str(uuid.uuid4()),
                tenant_id=str(uuid.uuid4()),
                parent_version_id=str(uuid.uuid4()),
                family="noise",
                intensity=0.5,
                seed=1,
            )
            handle = await env.client.start_workflow(
                PerturbationWorkflow.run, inp, id="pert-1", task_queue="pert"
            )
            uri = await handle.result()

    assert uri == "datasets/d/perturbations/j/artifact.parquet"
    assert calls == ["status:running", "apply", "register", "status:succeeded"]


def test_perturbation_input_defaults() -> None:
    inp = PerturbationInput(
        job_id="j",
        dataset_id="d",
        tenant_id="t",
        parent_version_id="p",
        family="noise",
        intensity=0.1,
        seed=0,
    )
    assert inp.modality == "tabular"
    assert inp.params == {}
    assert inp.target_fields == []
