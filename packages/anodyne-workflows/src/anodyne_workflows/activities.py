"""Real activity implementations for `GenerationWorkflow`.

These bind to infra (repo, object store, Ray) via a module-level context that
the worker sets once at startup (wired in Task 7 / `generation-worker`). Kept
thin on purpose: the workflow test (`tests/test_workflow.py`) exercises the
orchestration with mocked activities, not these implementations.

The four core activities (`plan_shards`/`generate_shards`/`assemble_and_upload`/
`register_version`) are modality-agnostic: each resolves the dataset's
`spec.modality` to a `ModalityHandler` via `anodyne_workflows.modality` and
delegates the modality-specific work there. Tabular is the default handler, so
the C0 path is byte-for-byte unchanged. Handlers self-register when
`anodyne_workflows.handlers` is imported (see the bottom of this module).
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from anodyne_core.models import ModelConfig
from anodyne_core.ports import ObjectStore, SecretStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob, JobStatus
from anodyne_dataset.ports import AudioProvider, DatasetRepository, ProfileRepository
from anodyne_storage.objectstore import S3ObjectStore
from temporalio import activity

from anodyne_workflows.modality import get_handler
from anodyne_workflows.workflow import GenerationInput

if TYPE_CHECKING:
    from anodyne_video.ports import VideoProvider, VideoProviderRegistry

# Rows per shard for the tabular `plan_shards` default. Keeps Ray tasks small
# enough to parallelize without so many shards that per-task overhead
# dominates. Other modalities set their own via their `ModalityHandler`.
_SHARD_ROWS = 50_000


class ProgressPublisher(Protocol):
    """Duck-typed sink for live progress (bound to Redis pub/sub by the worker)."""

    async def publish(self, channel: str, message: str) -> None: ...


class ModelRegistryLike(Protocol):
    """Structural type for the tenant model registry text generation needs.

    `anodyne_llm.registry.SqlModelRegistry` satisfies this in production
    (only `.get` is used here); tests substitute fakes.
    """

    async def get(self, tenant_id: uuid.UUID, config_id: uuid.UUID) -> ModelConfig | None: ...


class ImageConfigRegistry(Protocol):
    """Duck-typed per-tenant image-provider config lookup.

    `anodyne_image.registry.SqlImageProviderRegistry` is the real, DB-backed
    implementation (bound by the worker); tests substitute an in-memory fake.
    """

    async def list(self, tenant_id: uuid.UUID) -> list[ModelConfig]: ...


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

    All modality-specific fields below default to values that make tabular-only
    wiring (every existing call site/test) work unchanged; each is consulted
    only by its own `ModalityHandler`.
    """

    repo: DatasetRepository
    s3_bucket: str
    s3_client: Any
    publisher: ProgressPublisher | None = None
    # Tabular from-sample synthesis (see the tabular handler).
    profile_repo: ProfileRepository | None = None
    ctgan_epochs: int = 100
    enable_sdv: bool = False
    # Text generation.
    model_registry: ModelRegistryLike | None = None
    secret_key: str = ""
    # Image generation.
    image_registry: ImageConfigRegistry | None = None
    secret_store: SecretStore | None = None
    # Audio generation: resolves a Modality.AUDIO spec to the tenant's provider.
    audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] | None = None
    # Video generation.
    video_registry: VideoProviderRegistry | None = None
    video_providers: dict[str, VideoProvider] = field(default_factory=dict)


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


async def _spec_for(ctx: ActivityContext, inp: GenerationInput) -> DatasetSpec | None:
    return await ctx.repo.get_spec(uuid.UUID(inp.tenant_id), uuid.UUID(inp.dataset_id))


def _modality_of(spec: DatasetSpec | None) -> str:
    return str(spec.modality) if spec is not None else "tabular"


@activity.defn(name="plan_shards")
async def plan_shards(inp: GenerationInput) -> list[list[int]]:
    """Split `target_rows` into contiguous [start, count] chunks.

    The chunk size is the modality handler's `shard_rows` (tabular's 50k by
    default). If the spec can't be resolved, falls back to the tabular size --
    the conservative, previously-only behavior.
    """
    # Tolerate an unconfigured context here (unlike the other activities): the
    # chunk size is a pure planning detail, and `plan_shards` is unit-tested as
    # a standalone async function. Falls back to the tabular size when the spec
    # (or context) can't be resolved -- the conservative, previously-only size.
    shard_rows = _SHARD_ROWS
    if _ctx is not None:
        spec = await _spec_for(_ctx, inp)
        if spec is not None:
            shard_rows = get_handler(_modality_of(spec)).shard_rows

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
    """Generate each shard and upload it to the object store; return its keys.

    Dispatches to the dataset modality's `ModalityHandler` -- tabular (the
    default) generates Parquet shards on Ray exactly as before.
    """
    ctx = _context()
    store = _object_store(inp)
    spec = await _spec_for(ctx, inp)
    if spec is None:
        raise ValueError(f"dataset {inp.dataset_id} not found for tenant {inp.tenant_id}")
    return await get_handler(_modality_of(spec)).generate_shards(ctx, inp, spec, shards, store)


@activity.defn(name="assemble_and_upload")
async def assemble_and_upload(inp: GenerationInput, keys: list[str]) -> str:
    """Assemble the shard outputs into one artifact and upload it.

    Returns the durable object-store *key* (not a presigned URL): the key is
    what gets persisted as `DatasetVersion.artifact_uri`, and presigning only
    happens later, at download time, in the gateway. The concrete artifact
    shape (Parquet, JSONL, manifest, ...) is the modality handler's concern.
    """
    ctx = _context()
    store = _object_store(inp)
    spec = await _spec_for(ctx, inp)
    return await get_handler(_modality_of(spec)).assemble(ctx, inp, spec, keys, store)


@activity.defn(name="register_version")
async def register_version(inp: GenerationInput, uri: str, rows: int) -> None:
    """Record the generated artifact as a new `DatasetVersion`.

    The artifact `format` is the modality handler's `artifact_format`
    (tabular: `parquet`).
    """
    ctx = _context()
    spec = await _spec_for(ctx, inp)
    fmt = get_handler(_modality_of(spec)).artifact_format
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


# Import handlers for their registration side effect -- this is what populates
# the modality registry that the activities above dispatch through. Deferred to
# the bottom so `ActivityContext` and the helpers the handlers import are
# already defined (avoids an import cycle).
from anodyne_workflows import handlers as _handlers  # noqa: E402,F401
