import uuid
from typing import Any

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


async def test_video_workflow_runs_video_activity_sequence_not_tabular() -> None:
    """`modality="video"` must dispatch the video activities, never the tabular ones."""
    calls: list[str] = []

    @activity.defn(name="plan_video_items")
    async def plan_video_items(inp: GenerationInput) -> list[list[int]]:
        calls.append("plan_video")
        return [[0, 2], [2, 1]]

    @activity.defn(name="generate_video_items")
    async def generate_video_items(
        inp: GenerationInput, shards: list[list[int]]
    ) -> list[dict[str, Any]]:
        calls.append("gen_video")
        return [{"index": 0}, {"index": 1}, {"index": 2}]

    @activity.defn(name="assemble_video_manifest")
    async def assemble_video_manifest(inp: GenerationInput, items: list[dict[str, Any]]) -> str:
        calls.append("assemble_video")
        return "datasets/d/j/manifest.json"

    @activity.defn(name="register_video_version")
    async def register_video_version(inp: GenerationInput, uri: str, rows: int) -> None:
        calls.append("register_video")

    @activity.defn(name="set_status")
    async def set_status(inp: GenerationInput, status: str, progress: float) -> None: ...

    @activity.defn(name="plan_shards")
    async def plan_shards(inp: GenerationInput) -> list[list[int]]:
        calls.append("plan_tabular")  # should never run for a video job
        return [[0, 1]]

    async with await WorkflowEnvironment.start_time_skipping() as env:
        async with Worker(
            env.client,
            task_queue="gen-video",
            workflows=[GenerationWorkflow],
            activities=[
                plan_video_items,
                generate_video_items,
                assemble_video_manifest,
                register_video_version,
                set_status,
                plan_shards,
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
    assert calls == ["plan_video", "gen_video", "assemble_video", "register_video"]


def test_generation_input_defaults_modality_to_tabular() -> None:
    inp = GenerationInput(job_id="j", dataset_id="d", tenant_id="t", target_rows=1, seed=0)
    assert inp.modality == "tabular"
