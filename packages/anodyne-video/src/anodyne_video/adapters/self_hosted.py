"""Self-hosted, GPU-backed `VideoProvider` adapter.

`SelfHostedVideoProvider` wraps an injected synchronous "model runner" --
`Callable[[VideoGenerationRequest], bytes]` -- executed off the event loop via
`asyncio.to_thread`. In production this runner is the seam a Ray-GPU-backed
text-to-video model plugs into: construct it with something like

    provider = SelfHostedVideoProvider(
        runner=lambda req: ray.get(actor_handle.generate.remote(req)),
    )

where `actor_handle` is a Ray actor that has loaded a text-to-video model's
weights onto a GPU once and serves `generate` calls (e.g. a
Stable-Video-Diffusion-family or ModelScope-T2V actor, dispatched the same way
`anodyne_compute.ray_tasks.remote_generate_shard` dispatches tabular shard
generation). Building that actor -- real model weights + GPU scheduling -- is
explicitly out of scope here: no GPU is available in this environment. Tests
inject a plain fake runner instead.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from anodyne_video.models import VideoAsset, VideoGenerationRequest, VideoProviderConfig
from anodyne_video.ports import VideoProvider


class SelfHostedVideoProvider(VideoProvider):
    def __init__(self, runner: Callable[[VideoGenerationRequest], bytes]) -> None:
        self._runner = runner

    async def generate(
        self, config: VideoProviderConfig, request: VideoGenerationRequest
    ) -> VideoAsset:
        content = await asyncio.to_thread(self._runner, request)
        return VideoAsset(
            content=content,
            duration_seconds=request.duration_seconds,
            width=request.width,
            height=request.height,
            fps=request.fps,
            seed=request.seed,
            provider=config.provider,
            model=config.model,
        )
