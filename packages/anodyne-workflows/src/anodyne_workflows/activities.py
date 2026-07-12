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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_audio.generator import AudioDatasetGenerator
from anodyne_audio.models import AudioManifestItem
from anodyne_compute import remote_generate_shard
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob, JobStatus, Modality
from anodyne_dataset.ports import AudioProvider, DatasetRepository
from anodyne_storage.objectstore import S3ObjectStore
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
    # Resolves a `Modality.AUDIO` DatasetSpec to the tenant's configured
    # `AudioProvider` (see `generation_worker.audio.AudioProviderFactory`).
    # `None` in tabular-only wiring/tests -- see `_generate_audio_shards`.
    audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] | None = None


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


def _audio_item_key(inp: GenerationInput, index: int, fmt: str) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/audio/item-{index}.{fmt}"


def _audio_manifest_shard_key(inp: GenerationInput, index: int) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/audio/manifest-shard-{index}.json"


def _audio_manifest_key(inp: GenerationInput) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"


async def _generate_audio_shards(
    ctx: ActivityContext,
    inp: GenerationInput,
    spec: DatasetSpec,
    shards: list[list[int]],
    store: ObjectStore,
) -> list[str]:
    """Audio counterpart to the tabular Ray-shard loop in `generate_shards`.

    Each shard synthesizes its items via the tenant's `AudioProvider`
    (injected through `ctx.audio_provider_factory`), uploads every clip under
    its own object key, and uploads a small manifest *fragment* (JSON list of
    `AudioManifestItem`) for the shard -- mirroring how the tabular path
    uploads one Parquet fragment per shard. `assemble_and_upload` later merges
    these fragments into the final `manifest.json`.
    """
    if ctx.audio_provider_factory is None:
        raise RuntimeError(
            "no audio_provider_factory configured for audio generation; "
            "see ActivityContext.audio_provider_factory"
        )
    provider = await ctx.audio_provider_factory(spec)
    generator = AudioDatasetGenerator(provider)

    keys: list[str] = []
    for i, (start, count) in enumerate(shards):
        pairs = await generator.generate(spec, start, count, inp.seed)
        manifest_items: list[AudioManifestItem] = []
        for plan, result in pairs:
            item_key = _audio_item_key(inp, plan.index, result.format)
            await store.put(item_key, result.audio_bytes)
            manifest_items.append(
                AudioManifestItem(
                    index=plan.index,
                    object_key=item_key,
                    text=plan.request.text,
                    label=plan.label,
                    voice=plan.request.voice,
                    format=result.format,
                    duration_seconds=result.duration_seconds,
                )
            )
        shard_key = _audio_manifest_shard_key(inp, i)
        payload = json.dumps([m.model_dump(mode="json") for m in manifest_items])
        await store.put(shard_key, payload.encode())
        keys.append(shard_key)
    return keys


async def _assemble_audio_manifest(
    inp: GenerationInput, keys: list[str], store: ObjectStore
) -> str:
    """Merge per-shard manifest fragments into the final `manifest.json`."""
    items: list[dict[str, Any]] = []
    for key in keys:
        data = await store.get(key)
        items.extend(json.loads(data.decode()))
    items.sort(key=lambda d: d["index"])

    manifest = {"dataset_id": inp.dataset_id, "job_id": inp.job_id, "items": items}
    artifact_key = _audio_manifest_key(inp)
    await store.put(artifact_key, json.dumps(manifest).encode())
    return artifact_key


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

    # `Generator` is selected by `spec.modality` here: audio gets its own
    # provider-driven path; every other modality (today: tabular) keeps using
    # the Ray/TabularSampler loop below unchanged.
    if spec.modality is Modality.AUDIO:
        return await _generate_audio_shards(ctx, inp, spec, shards, store)

    keys: list[str] = []
    for i, (start, count) in enumerate(shards):
        ref = remote_generate_shard.remote(spec, start, count, inp.seed + i)
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
    ctx = _context()
    store = _object_store(inp)
    spec = await ctx.repo.get_spec(uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id))
    if spec is not None and spec.modality is Modality.AUDIO:
        return await _assemble_audio_manifest(inp, keys, store)

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
    spec = await ctx.repo.get_spec(uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id))
    fmt = "audio_manifest" if spec is not None and spec.modality is Modality.AUDIO else "parquet"
    version = DatasetVersion(
        id=uuid.uuid4(),
        tenant_id=uuid.UUID(inp.tenant_id),
        dataset_id=uuid.UUID(inp.dataset_id),
        artifact_uri=uri,
        format=fmt,
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
