from __future__ import annotations

from uuid import uuid4

from anodyne_audio.generator import AudioDatasetGenerator
from anodyne_dataset.models import (
    AudioSynthesisRequest,
    AudioSynthesisResult,
    DatasetSpec,
    FieldSpec,
    Modality,
    SemanticType,
)
from anodyne_dataset.ports import AudioProvider


class _MockProvider(AudioProvider):
    def __init__(self) -> None:
        self.calls: list[AudioSynthesisRequest] = []

    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        self.calls.append(request)
        return AudioSynthesisResult(
            audio_bytes=request.text.encode(), format="wav", duration_seconds=1.0
        )


def _spec(directives: dict[str, object] | None = None, rows: int = 5) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="",
        modality=Modality.AUDIO,
        source="description",
        fields=[FieldSpec(name="transcript", semantic_type=SemanticType.TEXT)],
        target_rows=rows,
        directives=directives or {},
    )


async def test_uses_explicit_prompts_and_labels() -> None:
    spec = _spec(
        {
            "audio": {
                "prompts": ["hi", "bye"],
                "labels": ["greeting", "farewell"],
                "voice": "v1",
            }
        }
    )
    provider = _MockProvider()

    pairs = await AudioDatasetGenerator(provider).generate(spec, 0, 2, seed=1)

    assert [p.request.text for p, _ in pairs] == ["hi", "bye"]
    assert [p.label for p, _ in pairs] == ["greeting", "farewell"]
    assert all(p.request.voice == "v1" for p, _ in pairs)
    assert [r.audio_bytes for _, r in pairs] == [b"hi", b"bye"]


async def test_falls_back_to_deterministic_text_without_prompts() -> None:
    spec = _spec(rows=3)

    a = await AudioDatasetGenerator(_MockProvider()).generate(spec, 0, 3, seed=7)
    b = await AudioDatasetGenerator(_MockProvider()).generate(spec, 0, 3, seed=7)

    assert [p.request.text for p, _ in a] == [p.request.text for p, _ in b]
    assert all(p.request.text for p, _ in a)


async def test_disjoint_shard_ranges_index_correctly() -> None:
    spec = _spec({"audio": {"prompts": [f"t{i}" for i in range(10)]}}, rows=10)

    pairs = await AudioDatasetGenerator(_MockProvider()).generate(spec, 5, 3, seed=1)

    assert [p.index for p, _ in pairs] == [5, 6, 7]
    assert [p.request.text for p, _ in pairs] == ["t5", "t6", "t7"]


async def test_calls_provider_once_per_item() -> None:
    provider = _MockProvider()
    spec = _spec(rows=4)

    await AudioDatasetGenerator(provider).generate(spec, 0, 4, seed=1)

    assert len(provider.calls) == 4


def test_plan_items_without_labels_or_voice_leaves_them_none() -> None:
    spec = _spec(rows=2)

    plans = AudioDatasetGenerator(_MockProvider()).plan_items(spec, 0, 2, seed=3)

    assert [p.label for p in plans] == [None, None]
    assert all(p.request.voice is None and p.request.language is None for p in plans)
