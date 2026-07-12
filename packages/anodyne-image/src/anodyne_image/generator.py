from __future__ import annotations

import asyncio

import pyarrow as pa  # type: ignore[import-untyped]
from anodyne_dataset.models import DatasetSpec
from anodyne_dataset.ports import Generator

from anodyne_image.models import GeneratedImage, ImagePromptItem
from anodyne_image.ports import ImageProvider
from anodyne_image.prompts import ImagePromptBuilder


class ImageGenerator(Generator):
    """Implements the shared `Generator` port for `Modality.IMAGE` -- the
    worker selects this (vs. `TabularSampler`, etc.) by `spec.modality`.

    Binds one `ImageProvider` (already bound to a tenant's model/key -- see
    `anodyne_image.factory.resolve_image_provider`) and drives it once per
    prompt in the shard. `Generator.generate` is a synchronous port method;
    `ImageProvider.generate` is async (network/GPU call), so this runs the
    per-item calls via `asyncio.run` -- safe because this always executes
    inside a plain (non-async) Ray remote task process, never inside an
    already-running event loop.

    Returns a `pyarrow.Table` shaped like `TabularSampler`'s output (one row
    per item) so it flows through the exact same shard-bytes -> object-store
    -> assemble pipeline C0 built for tabular shards; `assemble_and_upload`'s
    image branch unpacks `image_bytes` into individual files + a manifest.
    """

    def __init__(
        self,
        provider: ImageProvider,
        prompt_builder: ImagePromptBuilder | None = None,
        size: str = "1024x1024",
    ) -> None:
        self._provider = provider
        self._prompts = prompt_builder or ImagePromptBuilder()
        self._size = size

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        items = self._prompts.build(spec, start_row, count)
        images = asyncio.run(self._generate_all(items, seed))
        return pa.table(
            {
                "item_index": [item.item_index for item in items],
                "label": [item.label for item in items],
                "prompt": [item.prompt for item in items],
                "image_bytes": pa.array([img.data for img in images], type=pa.binary()),
                "mime_type": [img.mime_type for img in images],
            }
        )

    async def _generate_all(self, items: list[ImagePromptItem], seed: int) -> list[GeneratedImage]:
        return [
            await self._provider.generate(item.prompt, seed=seed + item.item_index, size=self._size)
            for item in items
        ]
