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
from anodyne_compute import remote_generate_shard, remote_generate_text_shard
from anodyne_core.models import ModelConfig
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob, JobStatus, Modality
from anodyne_dataset.ports import DatasetRepository
from anodyne_storage.objectstore import S3ObjectStore
from temporalio import activity

from anodyne_workflows.workflow import GenerationInput

# Rows per shard for `plan_shards`. Keeps Ray tasks small enough to parallelize
# without so many shards that per-task overhead dominates.
_SHARD_ROWS = 50_000
# Text shards are much smaller: each row costs an LLM call (batched), unlike
# tabular's local Faker sampling, so a shard should be cheap to retry/re-run.
_TEXT_SHARD_ROWS = 200


class ProgressPublisher(Protocol):
    """Duck-typed sink for live progress (bound to Redis pub/sub by the worker)."""

    async def publish(self, channel: str, message: str) -> None: ...


class ModelRegistryLike(Protocol):
    """Structural type for the tenant model registry text generation needs.

    `anodyne_llm.registry.SqlModelRegistry` satisfies this in production
    (only `.get` is used here); tests substitute fakes.
    """

    async def get(self, tenant_id: uuid.UUID, config_id: uuid.UUID) -> ModelConfig | None: ...


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

    `model_registry`/`secret_key` are only consulted for `modality="text"`
    datasets (see `generate_shards`); both default to values that make
    tabular-only wiring (existing call sites) work unchanged.
    """

    repo: DatasetRepository
    s3_bucket: str
    s3_client: Any
    publisher: ProgressPublisher | None = None
    model_registry: ModelRegistryLike | None = None
    secret_key: str = ""


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


def _artifact_key(inp: GenerationInput, ext: str) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.{ext}"


def _manifest_key(inp: GenerationInput) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"


async def _resolve_model_config(ctx: ActivityContext, inp: GenerationInput) -> ModelConfig:
    """Look up the tenant's chosen model for text generation.

    The `ModelConfig` returned still carries its `secret_ref` *encrypted* --
    see `anodyne_compute.ray_tasks_text` for why that's safe to pass to Ray.
    """
    if ctx.model_registry is None or inp.model_config_id is None:
        raise ValueError(
            "text generation requires a registered model: no model_registry/model_config_id "
            "configured for this activity context"
        )
    model_config = await ctx.model_registry.get(
        uuid.UUID(inp.tenant_id), uuid.UUID(inp.model_config_id)
    )
    if model_config is None:
        raise ValueError(f"model config {inp.model_config_id} not found for tenant {inp.tenant_id}")
    return model_config


@activity.defn(name="plan_shards")
async def plan_shards(inp: GenerationInput) -> list[list[int]]:
    """Split `target_rows` into contiguous [start, count] chunks.

    Chunk size depends on the dataset's modality (text shards are much
    smaller than tabular's, see `_TEXT_SHARD_ROWS`). If the spec can't be
    resolved for some reason, falls back to the tabular chunk size -- the
    conservative, previously-only behavior -- rather than guessing.
    """
    ctx = _context()
    spec = await ctx.repo.get_spec(uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id))
    shard_rows = (
        _TEXT_SHARD_ROWS if spec is not None and spec.modality == Modality.TEXT else _SHARD_ROWS
    )

    shards: list[list[int]] = []
    start = 0
    remaining = inp.target_rows
    while remaining > 0:
        count = min(shard_rows, remaining)
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

    keys: list[str] = []
    for i, (start, count) in enumerate(shards):
        if spec.modality == Modality.TEXT:
            model_config = await _resolve_model_config(ctx, inp)
            ref = remote_generate_text_shard.remote(
                spec, model_config, ctx.secret_key, start, count, inp.seed + i
            )
        else:
            ref = remote_generate_shard.remote(spec, start, count, inp.seed + i)
        data: bytes = await asyncio.to_thread(ray.get, ref)
        key = _shard_key(inp, i)
        await store.put(key, data)
        keys.append(key)
    return keys


async def _assemble_text_artifact(
    store: ObjectStore, inp: GenerationInput, spec: DatasetSpec, table: pa.Table
) -> str:
    """Write the concatenated table as JSONL + a sibling manifest; return the JSONL key."""
    rows = table.to_pylist()
    jsonl_bytes = "\n".join(json.dumps(row) for row in rows).encode()
    artifact_key = _artifact_key(inp, "jsonl")
    await store.put(artifact_key, jsonl_bytes)

    manifest = {
        "modality": "text",
        "dataset_id": inp.dataset_id,
        "job_id": inp.job_id,
        "fields": [f.name for f in spec.fields],
        "rows_produced": table.num_rows,
        "model_config_id": inp.model_config_id,
        "seed": inp.seed,
    }
    await store.put(_manifest_key(inp), json.dumps(manifest).encode())
    return artifact_key


@activity.defn(name="assemble_and_upload")
async def assemble_and_upload(inp: GenerationInput, keys: list[str]) -> str:
    """Concatenate shard Parquet tables into one artifact and upload it.

    Returns the durable object-store *key* (not a presigned URL): the key is
    what gets persisted as `DatasetVersion.artifact_uri`, and presigning only
    happens later, at download time, in the gateway.

    Shard bytes are always Parquet-encoded internally (a shared implementation
    detail, not the public artifact format -- see `anodyne_compute`). Text
    datasets are re-serialized here to JSONL + a manifest at the final-write
    step; tabular datasets keep writing a single concatenated Parquet file,
    byte-for-byte the same as before this branch was added.
    """
    ctx = _context()
    store = _object_store(inp)
    spec = await ctx.repo.get_spec(uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id))

    tables = []
    for key in keys:
        data = await store.get(key)
        tables.append(pq.read_table(io.BytesIO(data)))
    table = pa.concat_tables(tables) if tables else pa.table({})

    if spec is not None and spec.modality == Modality.TEXT:
        return await _assemble_text_artifact(store, inp, spec, table)

    buf = io.BytesIO()
    pq.write_table(table, buf)
    artifact_key = _artifact_key(inp, "parquet")
    await store.put(artifact_key, buf.getvalue())
    return artifact_key


@activity.defn(name="register_version")
async def register_version(inp: GenerationInput, uri: str, rows: int) -> None:
    """Record the generated artifact as a new `DatasetVersion`.

    Format is inferred from the artifact key's extension -- `assemble_and_upload`
    is the single place that decides JSONL-vs-Parquet, so this stays a cheap,
    context-free inference rather than a second spec lookup.
    """
    ctx = _context()
    version = DatasetVersion(
        id=uuid.uuid4(),
        tenant_id=uuid.UUID(inp.tenant_id),
        dataset_id=uuid.UUID(inp.dataset_id),
        artifact_uri=uri,
        format="jsonl" if uri.endswith(".jsonl") else "parquet",
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
