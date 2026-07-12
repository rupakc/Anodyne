"""Video-modality Temporal activities for `GenerationWorkflow`.

Deliberately a separate module from `activities.py`/`ActivityContext`: this
keeps the tabular pipeline (five original activities, its own context) fully
untouched by the video path, so the two can be developed, merged, and
reasoned about independently (see the C5 design doc's "why video doesn't
reuse the tabular `Generator` port" section). Bound to infra by the worker at
startup via `configure_video_activities`, exactly like `activities.py`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetVersion
from anodyne_dataset.ports import DatasetRepository
from anodyne_storage.objectstore import S3ObjectStore
from anodyne_video.generator import VideoDatasetGenerator
from anodyne_video.models import VideoManifest, VideoManifestItem
from anodyne_video.ports import VideoProvider, VideoProviderRegistry
from temporalio import activity

from anodyne_workflows.workflow import GenerationInput

# Clips are heavy (network- or GPU-bound, one file each) -- unlike tabular's
# 50k-row shards, a video "shard" batches only a handful of items so no
# single activity call runs too long.
_VIDEO_SHARD_ITEMS = 4


@dataclass
class VideoActivityContext:
    """Infra bound to the video activities by the worker at startup."""

    repo: DatasetRepository
    s3_bucket: str
    s3_client: Any
    video_registry: VideoProviderRegistry
    providers: dict[str, VideoProvider] = field(default_factory=dict)


_ctx: VideoActivityContext | None = None


def configure_video_activities(ctx: VideoActivityContext) -> None:
    """Bind these activities to infra. Called once by the worker at startup."""
    global _ctx
    _ctx = ctx


def _context() -> VideoActivityContext:
    if _ctx is None:
        raise RuntimeError(
            "anodyne_workflows.video_activities not configured: "
            "call configure_video_activities() first"
        )
    return _ctx


def _object_store(inp: GenerationInput, ctx: VideoActivityContext) -> ObjectStore:
    return S3ObjectStore(ctx.s3_bucket, uuid.UUID(inp.tenant_id), client=ctx.s3_client)


def _video_key(inp: GenerationInput, index: int) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/videos/item-{index}.mp4"


def _manifest_key(inp: GenerationInput) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"


@activity.defn(name="plan_video_items")
async def plan_video_items(inp: GenerationInput) -> list[list[int]]:
    """Split `target_rows` (item count, for video jobs) into contiguous shards."""
    shards: list[list[int]] = []
    start = 0
    remaining = inp.target_rows
    while remaining > 0:
        count = min(_VIDEO_SHARD_ITEMS, remaining)
        shards.append([start, count])
        start += count
        remaining -= count
    return shards or [[0, 0]]


@activity.defn(name="generate_video_items")
async def generate_video_items(
    inp: GenerationInput, shards: list[list[int]]
) -> list[dict[str, Any]]:
    """Generate each shard's clips, upload them, and return manifest-item dicts.

    Resolves the tenant's first *enabled* `VideoProviderConfig` (directive-
    driven provider selection is a follow-up -- see the design doc's "out of
    scope"). Raw clip bytes never leave this activity: only JSON-serializable
    manifest metadata (including the now-known `object_key`) is returned.
    """
    ctx = _context()
    tenant_id = uuid.UUID(inp.tenant_id)
    dataset_id = uuid.UUID(inp.dataset_id)

    spec = await ctx.repo.get_spec(tenant_id, dataset_id)
    if spec is None:
        raise ValueError(f"dataset {inp.dataset_id} not found for tenant {inp.tenant_id}")

    configs = [c for c in await ctx.video_registry.list(tenant_id) if c.enabled]
    if not configs:
        raise ValueError(f"no enabled video provider configured for tenant {inp.tenant_id}")
    config = configs[0]

    provider = ctx.providers.get(config.provider)
    if provider is None:
        raise ValueError(f"no VideoProvider adapter registered for provider {config.provider!r}")

    store = _object_store(inp, ctx)
    generator = VideoDatasetGenerator()

    items: list[dict[str, Any]] = []
    for start, count in shards:
        results = await generator.generate_items(
            spec, provider=provider, config=config, start_index=start, count=count, seed=inp.seed
        )
        for item, content in results:
            key = _video_key(inp, item.index)
            await store.put(key, content)
            items.append(item.model_copy(update={"object_key": key}).model_dump(mode="json"))
    return items


@activity.defn(name="assemble_video_manifest")
async def assemble_video_manifest(inp: GenerationInput, items: list[dict[str, Any]]) -> str:
    """Build the `VideoManifest` from generated items and upload it as JSON."""
    ctx = _context()
    manifest = VideoManifest(
        tenant_id=uuid.UUID(inp.tenant_id),
        dataset_id=uuid.UUID(inp.dataset_id),
        job_id=uuid.UUID(inp.job_id),
        items=[VideoManifestItem.model_validate(i) for i in items],
    )
    store = _object_store(inp, ctx)
    key = _manifest_key(inp)
    await store.put(key, manifest.model_dump_json().encode())
    return key


@activity.defn(name="register_video_version")
async def register_video_version(inp: GenerationInput, uri: str, rows: int) -> None:
    """Record the generated manifest as a new `DatasetVersion` (format `video-manifest`)."""
    ctx = _context()
    version = DatasetVersion(
        id=uuid.uuid4(),
        tenant_id=uuid.UUID(inp.tenant_id),
        dataset_id=uuid.UUID(inp.dataset_id),
        artifact_uri=uri,
        format="video-manifest",
        row_count=rows,
    )
    await ctx.repo.add_version(version)
