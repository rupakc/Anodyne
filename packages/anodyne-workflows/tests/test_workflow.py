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
    async def set_status(inp: GenerationInput, status: str, progress: float) -> None:
        # Regression guard: Temporal only coerces an activity's args to their
        # typed form when the workflow passes as many args as the activity
        # declares parameters. A trailing defaulted param (args < params) makes
        # `inp` arrive as a raw dict and every real status activity crash on
        # `inp.tenant_id`, hanging the job. Assert the typed form here so a
        # reintroduced signature/arg-count mismatch fails loudly in CI.
        assert isinstance(inp, GenerationInput), f"expected GenerationInput, got {type(inp)}"

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


async def test_video_workflow_runs_shared_activity_sequence() -> None:
    """A `modality="video"` job runs the *same* modality-agnostic activity
    sequence as tabular: the workflow dispatches by activity NAME only, and the
    per-modality behaviour lives behind those names in the modality registry
    (exercised in `test_video_activities.py`). This is the converged design --
    there are no separate `*_video_*` workflow activities. The workflow must
    never inspect `modality` itself (keeps `workflow.py` import-free of modality
    packages for Temporal determinism)."""
    calls: list[str] = []

    @activity.defn(name="plan_shards")
    async def plan_shards(inp: GenerationInput) -> list[list[int]]:
        calls.append("plan")
        return [[0, 2], [2, 1]]

    @activity.defn(name="generate_shards")
    async def generate_shards(inp: GenerationInput, shards: list[list[int]]) -> list[str]:
        calls.append("gen")
        return ["k0", "k1"]

    @activity.defn(name="assemble_and_upload")
    async def assemble_and_upload(inp: GenerationInput, keys: list[str]) -> str:
        calls.append("assemble")
        return "datasets/d/j/manifest.json"

    @activity.defn(name="register_version")
    async def register_version(inp: GenerationInput, uri: str, rows: int) -> None:
        calls.append("register")

    @activity.defn(name="set_status")
    async def set_status(inp: GenerationInput, status: str, progress: float) -> None:
        # Regression guard: Temporal only coerces an activity's args to their
        # typed form when the workflow passes as many args as the activity
        # declares parameters. A trailing defaulted param (args < params) makes
        # `inp` arrive as a raw dict and every real status activity crash on
        # `inp.tenant_id`, hanging the job. Assert the typed form here so a
        # reintroduced signature/arg-count mismatch fails loudly in CI.
        assert isinstance(inp, GenerationInput), f"expected GenerationInput, got {type(inp)}"

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="gen-video",
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
                target_rows=3,
                seed=1,
                modality="video",
            )
            handle = await env.client.start_workflow(
                GenerationWorkflow.run, inp, id="wf-video-1", task_queue="gen-video"
            )
            await handle.signal(GenerationWorkflow.approve_schema)
            uri = await handle.result()
    assert uri == "datasets/d/j/manifest.json"
    assert calls == ["plan", "gen", "assemble", "register"]


def test_generation_input_defaults_modality_to_tabular() -> None:
    inp = GenerationInput(job_id="j", dataset_id="d", tenant_id="t", target_rows=1, seed=0)
    assert inp.modality == "tabular"
