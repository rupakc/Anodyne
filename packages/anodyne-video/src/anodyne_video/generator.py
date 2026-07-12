"""Turns a video `DatasetSpec` into prompts and drives `VideoProvider` calls.

The video analogue of `anodyne_dataset.ports.Generator` -- but async and
manifest-shaped rather than a synchronous `pyarrow.Table` producer (see the
design doc's "why video doesn't reuse the tabular `Generator` port" section).
Storage-agnostic on purpose: uploading clips to the object store is the
Temporal activity's job (`anodyne_workflows.video_activities`), not this
class's, so it stays unit-testable with a fake `VideoProvider` alone.
"""

from __future__ import annotations

from anodyne_dataset.models import DatasetSpec

from anodyne_video.models import VideoGenerationRequest, VideoManifestItem, VideoProviderConfig
from anodyne_video.ports import VideoProvider


def build_video_prompt(spec: DatasetSpec, index: int) -> str:
    """Deterministic per-item prompt from the spec's description + directives.

    Same `(spec, index)` always yields the same prompt (reproducible shards);
    varying `index` varies the prompt so a batch of clips isn't identical.
    """
    parts = [f"Item {index}: {spec.description}."]
    for key in sorted(spec.directives):
        value = spec.directives[key]
        parts.append(f"{key.capitalize()}: {value}.")
    return " ".join(parts)


class VideoDatasetGenerator:
    """Generates a contiguous range of video items for one dataset spec."""

    async def generate_items(
        self,
        spec: DatasetSpec,
        *,
        provider: VideoProvider,
        config: VideoProviderConfig,
        start_index: int,
        count: int,
        seed: int,
    ) -> list[tuple[VideoManifestItem, bytes]]:
        results: list[tuple[VideoManifestItem, bytes]] = []
        for offset in range(count):
            index = start_index + offset
            prompt = build_video_prompt(spec, index)
            # Independent, reproducible seed per (base seed, item index) --
            # mirrors TabularSampler's per-offset RNG seeding so shards stay
            # disjoint and deterministic.
            item_seed = seed * 1_000_003 + index
            request = VideoGenerationRequest(prompt=prompt, seed=item_seed)
            asset = await provider.generate(config, request)
            item = VideoManifestItem(
                index=index,
                prompt=prompt,
                duration_seconds=asset.duration_seconds,
                width=asset.width,
                height=asset.height,
                fps=asset.fps,
                seed=asset.seed,
                provider=asset.provider,
                model=asset.model,
                content_type=asset.content_type,
                byte_size=len(asset.content),
            )
            results.append((item, asset.content))
        return results
