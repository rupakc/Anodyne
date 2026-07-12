from __future__ import annotations

import asyncio
from typing import Any, Protocol

import ray

from anodyne_image.errors import ImageProviderError
from anodyne_image.models import GeneratedImage
from anodyne_image.ports import ImageProvider


class DiffusionPipeline(Protocol):
    """The shape of a GPU-resident text-to-image pipeline -- e.g. a loaded
    `diffusers.StableDiffusionXLPipeline` pinned to a GPU. Deliberately a
    plain synchronous callable (that's what real inference is); adapters
    drive it off the event loop via `asyncio.to_thread`.
    """

    def __call__(self, prompt: str, *, seed: int, size: str) -> bytes: ...


class SelfHostedSDXLProvider(ImageProvider):
    """Adapter over a self-hosted OSS diffusion model (e.g. SDXL) running on
    GPU compute, per the architecture decision to serve such models via Ray
    GPU actors.

    No GPU is available in this build/test environment: `pipeline` defaults
    to `None`, and `generate()` raises a clear `ImageProviderError` (not a
    crash reaching for a nonexistent GPU) telling the caller how to configure
    one. Live deployment injects a real pipeline callable -- see
    `RayGpuActorPipeline` for the Ray-GPU-actor wiring shape. Tests inject a
    plain stub callable.
    """

    def __init__(self, pipeline: DiffusionPipeline | None = None) -> None:
        self._pipeline = pipeline

    async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage:
        if self._pipeline is None:
            raise ImageProviderError(
                "no self-hosted diffusion pipeline configured; live image "
                "generation requires a GPU node running an OSS model (e.g. "
                "SDXL via diffusers) -- inject one via "
                "SelfHostedSDXLProvider(pipeline=...), e.g. a RayGpuActorPipeline "
                "wrapping a GPU actor handle"
            )
        data = await asyncio.to_thread(self._pipeline, prompt, seed=seed, size=size)
        return GeneratedImage(data=data, mime_type="image/png")


class RayGpuActorPipeline:
    """Adapts a Ray actor handle (wrapping a loaded GPU pipeline, exposing a
    `generate(prompt, seed, size)` method) into the `DiffusionPipeline`
    callable `SelfHostedSDXLProvider` expects.

    This is the concrete "served via Ray GPU actors" wiring the architecture
    calls for: deploy an actor pinned to a GPU node pool (`num_gpus=1`) that
    loads the model once and serves `generate` calls; pass its handle here.
    Requires a running Ray cluster/session at call time (`ray.get`); tests use
    local-mode Ray with a fake actor -- no GPU or model weights needed to
    prove the wiring.
    """

    def __init__(self, actor_handle: Any) -> None:
        self._actor = actor_handle

    def __call__(self, prompt: str, *, seed: int, size: str) -> bytes:
        result: bytes = ray.get(self._actor.generate.remote(prompt, seed, size))
        return result
