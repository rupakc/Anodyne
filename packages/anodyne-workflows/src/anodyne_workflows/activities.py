"""Real activity implementations for `GenerationWorkflow`.

These bind to infra (repo, object store, Ray) via a module-level context that
the worker sets once at startup (wired in Task 7 / `generation-worker`). Kept
thin on purpose: the workflow test (`tests/test_workflow.py`) exercises the
orchestration with mocked activities, not these implementations.
"""

from __future__ import annotations

import asyncio
import io
import json
import uuid
from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_compute import remote_generate_shard
from anodyne_compute.sample_tasks import remote_generate_shard_from_generator
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob, JobStatus
from anodyne_dataset.ports import DatasetRepository, ProfileRepository
from anodyne_storage.objectstore import S3ObjectStore
from anodyne_tabular.builder import build_tabular_generator
from anodyne_tabular.io import read_sample
from temporalio import activity

from anodyne_workflows.workflow import GenerationInput

# Rows per shard for `plan_shards`. Keeps Ray tasks small enough to parallelize
# without so many shards that per-task overhead dominates.
_SHARD_ROWS = 50_000


class ProgressPublisher(Protocol):
    """Duck-typed sink for live progress (bound to Redis pub/sub by the worker)."""

    async def publish(self, channel: str, message: str) -> None: ...


@dataclass
class ActivityContext:
    """Infra bound to these activities by the worker at startup.

    Carries the object-store *bucket name* and a boto3 client rather than one
    pre-built `ObjectStore` -- each activity that touches storage builds its
    own tenant-scoped `S3ObjectStore` via `_object_store(inp)`. A single
    pre-built store (constructed once, tenant-agnostic) is how the
    worker/gateway used to disagree on key layout: the worker's store
    prepended a fixed nil-UUID namespace while the gateway's presigner
    prepends the *real* tenant, so a key written at `{nil}/{tenant}/...`
    could never be found at `{tenant}/{tenant}/...`. Building the store
    per-activity from `inp.tenant_id` keeps both sides using the exact same
    prefix.
    """

    repo: DatasetRepository
    s3_bucket: str
    s3_client: Any
    publisher: ProgressPublisher | None = None
    # From-sample tabular synthesis (see `_generate_shards_from_sample`). All optional/
    # defaulted so every existing `ActivityContext(repo=..., s3_bucket=..., s3_client=...)`
    # construction keeps working unchanged.
    profile_repo: ProfileRepository | None = None
    ctgan_epochs: int = 100
    enable_sdv: bool = False


_ctx: ActivityContext | None = None


def configure_activities(ctx: ActivityContext) -> None:
    """Bind these activities to infra. Called once by the worker at startup."""
    global _ctx
    _ctx = ctx


def _context() -> ActivityContext:
    if _ctx is None:
        raise RuntimeError(
            "anodyne_workflows.activities not configured: call configure_activities() first"
        )
    return _ctx


def _object_store(inp: GenerationInput) -> ObjectStore:
    ctx = _context()
    return S3ObjectStore(ctx.s3_bucket, uuid.UUID(inp.tenant_id), client=ctx.s3_client)


def _shard_key(inp: GenerationInput, index: int) -> str:
    # Tenant-relative: `S3ObjectStore` prepends `{tenant_id}/` itself, so this
    # key must NOT repeat it (see `ActivityContext` docstring above).
    return f"datasets/{inp.dataset_id}/{inp.job_id}/shard-{index}.parquet"


def _artifact_key(inp: GenerationInput) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.parquet"


@activity.defn(name="plan_shards")
async def plan_shards(inp: GenerationInput) -> list[list[int]]:
    """Split `target_rows` into contiguous [start, count] chunks of `_SHARD_ROWS`."""
    shards: list[list[int]] = []
    start = 0
    remaining = inp.target_rows
    while remaining > 0:
        count = min(_SHARD_ROWS, remaining)
        shards.append([start, count])
        start += count
        remaining -= count
    return shards or [[0, 0]]


@activity.defn(name="generate_shards")
async def generate_shards(inp: GenerationInput, shards: list[list[int]]) -> list[str]:
    """Generate each shard on Ray and upload it to the object store; return its keys."""
    ctx = _context()
    store = _object_store(inp)
    spec = await ctx.repo.get_spec(uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id))
    if spec is None:
        raise ValueError(f"dataset {inp.dataset_id} not found for tenant {inp.tenant_id}")

    if spec.source == "sample":
        return await _generate_shards_from_sample(ctx, store, spec, inp, shards)

    keys: list[str] = []
    for i, (start, count) in enumerate(shards):
        ref = remote_generate_shard.remote(spec, start, count, inp.seed + i)
        data: bytes = await asyncio.to_thread(ray.get, ref)
        key = _shard_key(inp, i)
        await store.put(key, data)
        keys.append(key)
    return keys


async def _generate_shards_from_sample(
    ctx: ActivityContext,
    store: ObjectStore,
    spec: DatasetSpec,
    inp: GenerationInput,
    shards: list[list[int]],
) -> list[str]:
    """From-sample path: fit a tabular synthesizer once, then sample each shard on Ray.

    Fitting (not just sampling) happens once per generation job -- refitting a
    statistical/deep model per shard would be wasteful and would break the
    seed-determinism contract (see `anodyne_tabular`'s generators).
    """
    if ctx.profile_repo is None:
        raise RuntimeError(
            "ActivityContext.profile_repo not configured: cannot generate a "
            "source='sample' dataset"
        )
    tenant_id, dataset_id = uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id)
    profile = await ctx.profile_repo.get_profile(tenant_id, dataset_id)
    if profile is None:
        raise ValueError(
            f"dataset {inp.dataset_id} has source='sample' but no profile; "
            "upload a sample before generating"
        )
    sample_bytes = await store.get(profile.sample_uri)
    sample_df = await asyncio.to_thread(read_sample, sample_bytes, profile.sample_filename)
    generator = await asyncio.to_thread(
        build_tabular_generator,
        inp.method,
        profile,
        sample_df,
        epochs=ctx.ctgan_epochs,
        enable_sdv=ctx.enable_sdv,
    )

    keys: list[str] = []
    for i, (start, count) in enumerate(shards):
        ref = remote_generate_shard_from_generator.remote(
            generator, spec, start, count, inp.seed + i
        )
        data: bytes = await asyncio.to_thread(ray.get, ref)
        key = _shard_key(inp, i)
        await store.put(key, data)
        keys.append(key)
    return keys


@activity.defn(name="assemble_and_upload")
async def assemble_and_upload(inp: GenerationInput, keys: list[str]) -> str:
    """Concatenate shard Parquet tables into one artifact and upload it.

    Returns the durable object-store *key* (not a presigned URL): the key is
    what gets persisted as `DatasetVersion.artifact_uri`, and presigning only
    happens later, at download time, in the gateway.
    """
    store = _object_store(inp)
    tables = []
    for key in keys:
        data = await store.get(key)
        tables.append(pq.read_table(io.BytesIO(data)))

    table = pa.concat_tables(tables) if tables else pa.table({})
    buf = io.BytesIO()
    pq.write_table(table, buf)

    artifact_key = _artifact_key(inp)
    await store.put(artifact_key, buf.getvalue())
    return artifact_key


@activity.defn(name="register_version")
async def register_version(inp: GenerationInput, uri: str, rows: int) -> None:
    """Record the generated artifact as a new `DatasetVersion`."""
    ctx = _context()
    version = DatasetVersion(
        id=uuid.uuid4(),
        tenant_id=uuid.UUID(inp.tenant_id),
        dataset_id=uuid.UUID(inp.dataset_id),
        artifact_uri=uri,
        row_count=rows,
    )
    await ctx.repo.add_version(version)


@activity.defn(name="set_status")
async def set_status(
    inp: GenerationInput, status: str, progress: float, message: str | None = None
) -> None:
    """Update the `GenerationJob` status and publish live progress to Redis.

    `save_job` is a full-column upsert, so we must fetch the existing job and
    mutate it in place rather than constructing a fresh `GenerationJob` here —
    otherwise fields the gateway set at job creation (`workflow_id`, `message`)
    would be silently wiped on every status transition.
    """
    ctx = _context()
    tenant_id = uuid.UUID(inp.tenant_id)
    job_id = uuid.UUID(inp.job_id)
    job = await ctx.repo.get_job(tenant_id, job_id)
    if job is None:
        # Shouldn't happen in practice (the gateway creates the job up front),
        # but fall back to a minimal record rather than crashing the activity.
        job = GenerationJob(id=job_id, tenant_id=tenant_id, dataset_id=uuid.UUID(inp.dataset_id))
    job.status = JobStatus(status)
    job.progress = progress
    if message is not None:
        job.message = message
    await ctx.repo.save_job(job)
    if ctx.publisher is not None:
        message = json.dumps({"job_id": inp.job_id, "status": status, "progress": progress})
        await ctx.publisher.publish(f"job:{inp.job_id}", message)
