from uuid import UUID, uuid4

import pytest
from anodyne_video.models import VideoAsset, VideoGenerationRequest, VideoProviderConfig
from anodyne_video.ports import VideoProvider, VideoProviderRegistry


class _FakeProvider(VideoProvider):
    async def generate(
        self, config: VideoProviderConfig, request: VideoGenerationRequest
    ) -> VideoAsset:
        return VideoAsset(
            content=b"fake-mp4-bytes",
            duration_seconds=request.duration_seconds,
            width=request.width,
            height=request.height,
            fps=request.fps,
            seed=request.seed,
            provider=config.provider,
            model=config.model,
        )


class _FakeRegistry(VideoProviderRegistry):
    def __init__(self) -> None:
        self._configs: dict[UUID, VideoProviderConfig] = {}

    async def create(
        self,
        tenant_id: UUID,
        *,
        name: str,
        provider: str,
        model: str,
        api_key: str | None,
        api_base: str | None,
        params: dict[str, object],
    ) -> VideoProviderConfig:
        cfg = VideoProviderConfig(
            id=uuid4(),
            tenant_id=tenant_id,
            name=name,
            provider=provider,
            model=model,
            params=params,
            api_base=api_base,
        )
        self._configs[cfg.id] = cfg
        return cfg

    async def get(self, tenant_id: UUID, config_id: UUID) -> VideoProviderConfig | None:
        return self._configs.get(config_id)

    async def list(self, tenant_id: UUID) -> list[VideoProviderConfig]:
        return list(self._configs.values())

    async def delete(self, tenant_id: UUID, config_id: UUID) -> None:
        self._configs.pop(config_id, None)


async def test_fake_provider_conforms_to_port() -> None:
    provider: VideoProvider = _FakeProvider()
    cfg = VideoProviderConfig(id=uuid4(), tenant_id=uuid4(), name="c", provider="p", model="m")
    asset = await provider.generate(cfg, VideoGenerationRequest(prompt="x", seed=1))
    assert asset.content == b"fake-mp4-bytes"
    assert asset.provider == "p"


async def test_fake_registry_conforms_to_port() -> None:
    registry: VideoProviderRegistry = _FakeRegistry()
    tid = uuid4()
    cfg = await registry.create(
        tid, name="c", provider="p", model="m", api_key="k", api_base=None, params={}
    )
    assert await registry.get(tid, cfg.id) == cfg
    assert await registry.list(tid) == [cfg]
    await registry.delete(tid, cfg.id)
    assert await registry.get(tid, cfg.id) is None


def test_video_provider_is_abstract() -> None:
    with pytest.raises(TypeError):
        VideoProvider()  # type: ignore[abstract]


def test_video_provider_registry_is_abstract() -> None:
    with pytest.raises(TypeError):
        VideoProviderRegistry()  # type: ignore[abstract]
