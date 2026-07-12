import uuid

import pytest
from anodyne_workflows.workflow import GenerationInput, GenerationWorkflow
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

pytestmark = pytest.mark.integration  # needs the temporal test server download


async def test_workflow_runs_after_approval() -> None:
    calls: list[str] = []

    @activity.defn(name="plan_shards")
    async def plan_shards(inp: GenerationInput) -> list[list[int]]:
        calls.append("plan")
        return [[0, 5], [5, 5]]

    @activity.defn(name="generate_shards")
    async def generate_shards(inp: GenerationInput, shards: list[list[int]]) -> list[str]:
        calls.append("gen")
        return ["k0", "k1"]

    @activity.defn(name="assemble_and_upload")
    async def assemble_and_upload(inp: GenerationInput, keys: list[str]) -> str:
        calls.append("assemble")
        return "s3://bucket/artifact.parquet"

    @activity.defn(name="register_version")
    async def register_version(inp: GenerationInput, uri: str, rows: int) -> None:
        calls.append("register")

    @activity.defn(name="set_status")
    async def set_status(inp: GenerationInput, status: str, progress: float) -> None: ...

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="gen",
            workflows=[GenerationWorkflow],
            activities=[
                plan_shards,
                generate_shards,
                assemble_and_upload,
                register_version,
                set_status,
            ],
        ):
            inp = GenerationInput(
                job_id=str(uuid.uuid4()),
                dataset_id=str(uuid.uuid4()),
                tenant_id=str(uuid.uuid4()),
                target_rows=10,
                seed=1,
            )
            handle = await env.client.start_workflow(
                GenerationWorkflow.run, inp, id="wf-1", task_queue="gen"
            )
            await handle.signal(GenerationWorkflow.approve_schema)  # HITL gate
            uri = await handle.result()
    assert uri == "s3://bucket/artifact.parquet"
    assert calls == ["plan", "gen", "assemble", "register"]
