from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

from temporalio import workflow
from temporalio.common import RetryPolicy


@dataclass
class GenerationInput:
    job_id: str
    dataset_id: str
    tenant_id: str
    target_rows: int
    seed: int
    # Back-compat default: existing tabular call sites that don't pass this
    # keep working unchanged. `"video"` dispatches the video activity
    # sequence instead (see `GenerationWorkflow.run` below and
    # `anodyne_workflows.video_activities`).
    modality: str = "tabular"


@workflow.defn
class GenerationWorkflow:
    def __init__(self) -> None:
        self._approved = False

    @workflow.signal
    def approve_schema(self) -> None:
        self._approved = True

    @workflow.run
    async def run(self, inp: GenerationInput) -> str:
        opts: dict[str, Any] = dict(
            start_to_close_timeout=timedelta(minutes=10),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )
        await workflow.execute_activity("set_status", args=[inp, "awaiting_review", 0.0], **opts)
        await workflow.wait_condition(lambda: self._approved)  # HITL gate
        await workflow.execute_activity("set_status", args=[inp, "running", 0.1], **opts)

        # Per-modality activity sequence. Each modality's plan/generate/assemble
        # steps differ enough (row-shard-parquet for tabular vs.
        # item-shard-clip-manifest for video) that they get their own named
        # activities rather than forcing one shape on both; `set_status`
        # (progress) and the final artifact-URI-returning contract are shared.
        # Additional modalities extend this branch (see the C5 design doc).
        if inp.modality == "video":
            shards = await workflow.execute_activity("plan_video_items", args=[inp], **opts)
            items = await workflow.execute_activity(
                "generate_video_items", args=[inp, shards], **opts
            )
            await workflow.execute_activity("set_status", args=[inp, "running", 0.7], **opts)
            uri: str = await workflow.execute_activity(
                "assemble_video_manifest", args=[inp, items], **opts
            )
            await workflow.execute_activity(
                "register_video_version", args=[inp, uri, len(items)], **opts
            )
        else:
            shards = await workflow.execute_activity("plan_shards", args=[inp], **opts)
            keys = await workflow.execute_activity("generate_shards", args=[inp, shards], **opts)
            await workflow.execute_activity("set_status", args=[inp, "running", 0.7], **opts)
            uri = await workflow.execute_activity("assemble_and_upload", args=[inp, keys], **opts)
            await workflow.execute_activity(
                "register_version", args=[inp, uri, inp.target_rows], **opts
            )

        await workflow.execute_activity("set_status", args=[inp, "succeeded", 1.0], **opts)
        return uri
