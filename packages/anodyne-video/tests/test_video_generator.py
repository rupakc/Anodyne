from uuid import uuid4

from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_video.generator import VideoDatasetGenerator, build_video_prompt
from anodyne_video.models import VideoAsset, VideoGenerationRequest, VideoProviderConfig
from anodyne_video.ports import VideoProvider


def _spec(**overrides: object) -> DatasetSpec:
    defaults: dict[str, object] = dict(
        id=uuid4(),
        tenant_id=uuid4(),
        name="surf-cats",
        description="short clips of cats surfing",
        modality=Modality.VIDEO,
        source="description",
        fields=[],
        target_rows=5,
        directives={"style": "noir", "scene": "beach"},
    )
    defaults.update(overrides)
    return DatasetSpec.model_validate(defaults)


def test_build_video_prompt_is_deterministic_and_varies_by_index() -> None:
    spec = _spec()
    p0a = build_video_prompt(spec, 0)
    p0b = build_video_prompt(spec, 0)
    p1 = build_video_prompt(spec, 1)

    assert p0a == p0b
    assert p0a != p1
    assert spec.description in p0a
    assert "noir" in p0a
    assert "beach" in p0a


def test_build_video_prompt_omits_directives_key_when_absent() -> None:
    spec = _spec(directives={})
    prompt = build_video_prompt(spec, 0)
    assert spec.description in prompt


class _FakeProvider(VideoProvider):
    def __init__(self) -> None:
        self.calls: list[VideoGenerationRequest] = []

    async def generate(
        self, config: VideoProviderConfig, request: VideoGenerationRequest
    ) -> VideoAsset:
        self.calls.append(request)
        return VideoAsset(
            content=f"clip-{request.seed}".encode(),
            duration_seconds=request.duration_seconds,
            width=request.width,
            height=request.height,
            fps=request.fps,
            seed=request.seed,
            provider=config.provider,
            model=config.model,
        )


async def test_generate_items_returns_contiguous_items_with_matching_prompts() -> None:
    spec = _spec()
    provider = _FakeProvider()
    config = VideoProviderConfig(
        id=uuid4(), tenant_id=spec.tenant_id, name="c", provider="replicate", model="m"
    )
    generator = VideoDatasetGenerator()

    results = await generator.generate_items(
        spec, provider=provider, config=config, start_index=0, count=3, seed=42
    )

    assert [item.index for item, _ in results] == [0, 1, 2]
    for item, content in results:
        assert item.prompt == build_video_prompt(spec, item.index)
        assert content == f"clip-{item.seed}".encode()
        assert item.provider == "replicate"
        assert item.byte_size == len(content)
    assert len(provider.calls) == 3


async def test_generate_items_respects_start_index_offset() -> None:
    spec = _spec()
    provider = _FakeProvider()
    config = VideoProviderConfig(
        id=uuid4(), tenant_id=spec.tenant_id, name="c", provider="replicate", model="m"
    )
    generator = VideoDatasetGenerator()

    results = await generator.generate_items(
        spec, provider=provider, config=config, start_index=10, count=2, seed=1
    )

    assert [item.index for item, _ in results] == [10, 11]
