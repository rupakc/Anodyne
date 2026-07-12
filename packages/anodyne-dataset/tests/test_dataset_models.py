from uuid import uuid4

import pytest
from anodyne_dataset.models import (
    AudioSynthesisRequest,
    AudioSynthesisResult,
    DatasetSpec,
    FieldSpec,
    GenerationJob,
    JobStatus,
    Modality,
    SemanticType,
)
from anodyne_dataset.ports import AudioProvider


def test_fieldspec_defaults() -> None:
    f = FieldSpec(name="age", semantic_type=SemanticType.INTEGER)
    assert f.nullable is False and f.constraints == {}


def test_datasetspec_is_tabular_description() -> None:
    spec = DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="people",
        modality=Modality.TABULAR,
        source="description",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=100,
    )
    assert spec.status == "draft" and spec.fields[0].name == "age"


def test_job_progress_bounds() -> None:
    j = GenerationJob(id=uuid4(), tenant_id=uuid4(), dataset_id=uuid4())
    assert j.status is JobStatus.PENDING and j.progress == 0.0


def test_audio_synthesis_request_defaults() -> None:
    r = AudioSynthesisRequest(text="hello")
    assert r.voice is None and r.language is None


def test_audio_synthesis_result_defaults_to_wav() -> None:
    res = AudioSynthesisResult(audio_bytes=b"\x00\x01")
    assert res.format == "wav" and res.duration_seconds is None


async def test_audio_provider_is_an_abstract_async_contract() -> None:
    class _Echo(AudioProvider):
        async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
            return AudioSynthesisResult(audio_bytes=request.text.encode())

    out = await _Echo().synthesize(AudioSynthesisRequest(text="hi"))
    assert out.audio_bytes == b"hi"
    with pytest.raises(TypeError):
        AudioProvider()  # type: ignore[abstract]
