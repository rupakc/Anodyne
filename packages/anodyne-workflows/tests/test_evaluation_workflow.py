"""EvaluationWorkflow orchestration test (needs the temporal test server download).

Mirrors `test_workflow.py`: mocked activities record the call sequence; asserts
the workflow runs set_eval_status -> run_evaluation -> set_eval_status and
returns the report key.
"""

from __future__ import annotations

import uuid

import pytest
from anodyne_workflows.evaluation_workflow import EvaluationInput, EvaluationWorkflow
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

pytestmark = pytest.mark.integration


async def test_evaluation_workflow_runs_and_returns_report_key() -> None:
    calls: list[str] = []

    @activity.defn(name="set_eval_status")
    async def set_eval_status(inp: EvaluationInput, status: str, progress: float) -> None:
        calls.append(f"status:{status}")

    @activity.defn(name="run_evaluation")
    async def run_evaluation(inp: EvaluationInput) -> str:
        calls.append("run")
        return f"evaluations/{inp.run_id}/report.json"

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="eval",
            workflows=[EvaluationWorkflow],
            activities=[set_eval_status, run_evaluation],
        ):
            run_id = str(uuid.uuid4())
            inp = EvaluationInput(
                run_id=run_id,
                dataset_id=str(uuid.uuid4()),
                tenant_id=str(uuid.uuid4()),
                dataset_version_id=str(uuid.uuid4()),
                seed=1,
            )
            key = await env.client.execute_workflow(
                EvaluationWorkflow.run, inp, id="eval-wf-1", task_queue="eval"
            )
    assert key == f"evaluations/{run_id}/report.json"
    assert calls == ["status:running", "run", "status:succeeded"]


def test_evaluation_input_defaults() -> None:
    inp = EvaluationInput(run_id="r", dataset_id="d", tenant_id="t", dataset_version_id="v")
    assert inp.reference_version_id is None
    assert inp.config == {}
