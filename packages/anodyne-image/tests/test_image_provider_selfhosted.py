from __future__ import annotations

import pytest
from anodyne_image.errors import ImageProviderError
from anodyne_image.providers.selfhosted import SelfHostedSDXLProvider


class _StubPipeline:
    """Stand-in for a GPU-resident diffusers pipeline: records its call and
    returns fixed bytes -- no torch/diffusers/GPU involved."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def __call__(self, prompt: str, *, seed: int, size: str) -> bytes:
        self.calls.append({"prompt": prompt, "seed": seed, "size": size})
        return f"{prompt}:{seed}:{size}".encode()


async def test_generate_uses_injected_pipeline() -> None:
    pipeline = _StubPipeline()
    provider = SelfHostedSDXLProvider(pipeline=pipeline)

    result = await provider.generate("a mountain", seed=42, size="1024x1024")

    assert result.data == b"a mountain:42:1024x1024"
    assert result.mime_type == "image/png"
    assert pipeline.calls == [{"prompt": "a mountain", "seed": 42, "size": "1024x1024"}]


async def test_missing_pipeline_raises_clear_error() -> None:
    provider = SelfHostedSDXLProvider()

    with pytest.raises(ImageProviderError, match="GPU"):
        await provider.generate("a mountain", seed=1)
