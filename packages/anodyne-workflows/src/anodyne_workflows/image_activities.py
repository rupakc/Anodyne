"""Image-modality implementation for the shared shard/assemble activities in
`anodyne_workflows.activities`.

Kept in its own module (rather than growing inline in `activities.py`) so the
shared file's diff stays a small, easy-to-reconcile `if modality == "image"`
branch in each of `generate_shards`/`assemble_and_upload`/`register_version`
-- other modality specs (C2/C4/C5) are expected to add their own analogous
branches + modules there too.
"""

from __future__ import annotations

import asyncio
import io
import json
from typing import TYPE_CHECKING, Any

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import ray
from anodyne_compute.image_tasks import remote_generate_image_shard
from anodyne_core.models import ModelConfig
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec

if TYPE_CHECKING:
    from anodyne_workflows.workflow import GenerationInput


def _shard_key(inp: GenerationInput, index: int) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/image-shard-{index}.parquet"


def _image_object_key(inp: GenerationInput, item_index: int, mime_type: str) -> str:
    ext = mime_type.split("/")[-1] if "/" in mime_type else "bin"
    return f"datasets/{inp.dataset_id}/{inp.job_id}/images/{item_index}.{ext}"


def _manifest_key(inp: GenerationInput) -> str:
    return f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"


async def generate_image_shards(
    inp: GenerationInput,
    shards: list[list[int]],
    spec: DatasetSpec,
    store: ObjectStore,
    provider_config: ModelConfig,
    api_key: str | None,
) -> list[str]:
    """Ray-generate each shard's rows (item_index/label/prompt/image_bytes/
    mime_type) as a Parquet blob and upload it -- the image-modality analogue
    of `generate_shards`'s tabular path. Returns per-shard keys for
    `assemble_image_manifest` to unpack into individual image files.
    """
    keys: list[str] = []
    for i, (start, count) in enumerate(shards):
        ref = remote_generate_image_shard.remote(
            spec, start, count, inp.seed + i, provider_config, api_key
        )
        data: bytes = await asyncio.to_thread(ray.get, ref)
        key = _shard_key(inp, i)
        await store.put(key, data)
        keys.append(key)
    return keys


async def assemble_image_manifest(inp: GenerationInput, keys: list[str], store: ObjectStore) -> str:
    """Unpack every shard's rows into individual image objects + one
    `manifest.json` (`{"items": [{item_index, object_key, prompt, label,
    mime_type}, ...]}`) -- the final artifact for an image dataset version.
    """
    entries: list[dict[str, Any]] = []
    for key in keys:
        data = await store.get(key)
        table = pq.read_table(io.BytesIO(data))
        for row in table.to_pylist():
            object_key = _image_object_key(inp, row["item_index"], row["mime_type"])
            await store.put(object_key, row["image_bytes"])
            entries.append(
                {
                    "item_index": row["item_index"],
                    "object_key": object_key,
                    "prompt": row["prompt"],
                    "label": row["label"],
                    "mime_type": row["mime_type"],
                }
            )
    entries.sort(key=lambda e: int(e["item_index"]))
    manifest_key = _manifest_key(inp)
    await store.put(manifest_key, json.dumps({"items": entries}).encode())
    return manifest_key
