from uuid import uuid4

import pytest
from anodyne_video.adapters.self_hosted import SelfHostedVideoProvider
from anodyne_video.models import VideoGenerationRequest, VideoProviderConfig


def _config(**overrides: object) -> VideoProviderConfig:
    defaults: dict[str, object] = dict(
        id=uuid4(),
        tenant_id=uuid4(),
        name="local",
        provider="self-hosted",
        model="stable-video-diffusion",
    )
    defaults.update(overrides)
    return VideoProviderConfig.model_validate(defaults)


async def test_generate_calls_injected_runner_off_thread_and_wraps_asset() -> None:
    calls: list[VideoGenerationRequest] = []

    def runner(request: VideoGenerationRequest) -> bytes:
        calls.append(request)
        return b"fake-mp4-bytes"

    provider = SelfHostedVideoProvider(runner=runner)
    config = _config()
    request = VideoGenerationRequest(prompt="a cat surfing", seed=7, width=512, height=288, fps=6)

    asset = await provider.generate(config, request)

    assert calls == [request]
    assert asset.content == b"fake-mp4-bytes"
    assert asset.width == 512
    assert asset.height == 288
    assert asset.fps == 6
    assert asset.seed == 7
    assert asset.provider == "self-hosted"
    assert asset.model == "stable-video-diffusion"


async def test_generate_propagates_runner_exceptions() -> None:
    def failing_runner(request: VideoGenerationRequest) -> bytes:
        raise RuntimeError("no GPU available")

    provider = SelfHostedVideoProvider(runner=failing_runner)

    with pytest.raises(RuntimeError, match="no GPU available"):
        await provider.generate(_config(), VideoGenerationRequest(prompt="x", seed=1))
