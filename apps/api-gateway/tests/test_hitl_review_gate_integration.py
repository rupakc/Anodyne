"""Proves the generalized HITL review gate (`api_gateway.hitl_routes.
apply_review_decision`) actually resumes the real, unmodified
`GenerationWorkflow` -- not a stand-in. Requires Docker/the Temporal test
server (same requirement as `anodyne_workflows/tests/test_workflow.py`, which
this mirrors); not run in the default suite.

Unlike `test_workflow.py::test_workflow_runs_after_approval` (which signals
directly via `handle.signal(GenerationWorkflow.approve_schema)`), this starts
the workflow *without* `start_signal` so it truly parks at `wait_condition`,
then resumes it exclusively through the generic `ReviewTask` ->
`apply_review_decision` path any HITL-gated workflow would use -- proving the
gate generalizes without touching `GenerationWorkflow` or the existing eager
`start_signal="approve_schema"` behavior in `start_generation`
(`apps/api-gateway/src/api_gateway/app.py`), which stays untouched and is
covered by its own pre-existing tests.
"""

from __future__ import annotations

import uuid
from uuid import uuid4

import pytest
from anodyne_hitl.models import ReviewKind, ReviewStatus, ReviewTask
from anodyne_hitl.ports import ReviewRepository
from anodyne_workflows.workflow import GenerationInput, GenerationWorkflow
from api_gateway.hitl_routes import apply_review_decision
from temporalio import activity
from temporalio.testing import WorkflowEnvironment
from temporalio.worker import Worker

pytestmark = pytest.mark.integration


class _FakeReviewRepo(ReviewRepository):
    """Only `save` is exercised; the rest is a stub to satisfy the ABC so
    `apply_review_decision`'s `ReviewRepository` type hint is real, not a
    structural duck-typed fake (see `_FakeRepo` classes elsewhere)."""

    def __init__(self) -> None:
        self.saved: ReviewTask | None = None

    async def create(self, task: ReviewTask) -> None: ...

    async def get(self, tenant_id: uuid.UUID, review_id: uuid.UUID) -> ReviewTask | None:
        return None

    async def list_for_tenant(
        self, tenant_id: uuid.UUID, status: ReviewStatus | None = None
    ) -> list[ReviewTask]:
        return []

    async def save(self, task: ReviewTask) -> None:
        self.saved = task


async def test_approving_a_pending_review_task_resumes_the_real_generation_workflow() -> None:
    calls: list[str] = []

    @activity.defn(name="plan_shards")
    async def plan_shards(inp: GenerationInput) -> list[list[int]]:
        calls.append("plan")
        return [[0, 5]]

    @activity.defn(name="generate_shards")
    async def generate_shards(inp: GenerationInput, shards: list[list[int]]) -> list[str]:
        calls.append("gen")
        return ["k0"]

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
            task_queue="gen-hitl",
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
                target_rows=5,
                seed=1,
            )
            # No `start_signal` -- the workflow genuinely parks at
            # `wait_condition`, unlike the gateway's eager auto-approved path.
            handle = await env.client.start_workflow(
                GenerationWorkflow.run, inp, id="gen-hitl-1", task_queue="gen-hitl"
            )

            task = ReviewTask(
                id=uuid4(),
                tenant_id=uuid.UUID(inp.tenant_id),
                kind=ReviewKind.SCHEMA_APPROVAL,
                target_type="dataset",
                target_id=uuid.UUID(inp.dataset_id),
                workflow_id=handle.id,
                # signal_name intentionally omitted: resolved via
                # `default_signal_name(ReviewKind.SCHEMA_APPROVAL)`.
            )
            review_repo = _FakeReviewRepo()

            decided = await apply_review_decision(
                env.client, review_repo, task, "approve", comment=None
            )

            assert decided.status == ReviewStatus.APPROVED
            assert review_repo.saved is decided

            uri = await handle.result()

    assert uri == "s3://bucket/artifact.parquet"
    assert calls == ["plan", "gen", "assemble", "register"]
