"""Proves the self-hosted GPU wiring shape (`RayGpuActorPipeline` adapts a Ray
actor handle into the `DiffusionPipeline` callable `SelfHostedSDXLProvider`
expects) using an embedded single-process Ray instance -- no GPU, no
`diffusers`/`torch`, no cluster. Mirrors
`packages/anodyne-compute/tests/test_ray_tasks.py`'s Ray-integration
convention exactly.
"""

from __future__ import annotations

import pytest
import ray
from anodyne_image.providers.selfhosted import RayGpuActorPipeline, SelfHostedSDXLProvider

pytestmark = pytest.mark.integration


@ray.remote
class _FakeGpuActor:
    """Stands in for an actor wrapping a loaded SDXL pipeline pinned to a GPU:
    same `.generate.remote(prompt, seed, size)` shape, fixed bytes instead of
    real inference."""

    def generate(self, prompt: str, seed: int, size: str) -> bytes:
        return f"{prompt}:{seed}:{size}".encode()


async def test_ray_actor_pipeline_wiring() -> None:
    ray.init(ignore_reinit_error=True)
    try:
        actor = _FakeGpuActor.remote()  # type: ignore[attr-defined]
        provider = SelfHostedSDXLProvider(pipeline=RayGpuActorPipeline(actor))

        result = await provider.generate("a robot", seed=7, size="768x768")

        assert result.data == b"a robot:7:768x768"
    finally:
        ray.shutdown()
