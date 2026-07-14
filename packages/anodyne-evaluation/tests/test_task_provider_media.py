from __future__ import annotations

import json

# Importing the package registers every provider (media included) as a side
# effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import provider_for


class _FixedFakeProvider:
    """Fake `LLMProvider` that always returns a fixed JSON payload, regardless of
    prompt content. Used both for the boolean-array ("consistent") oracles and the
    single 1-5 rubric ("transcript_quality"/"prompt_quality") oracles -- a payload
    carrying both keys satisfies whichever one a given call parses."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.calls: list[LLMRequest] = []

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.calls.append(request)
        return LLMResponse(content=json.dumps(self._payload), usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


class _ExplodingProvider:
    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        raise AssertionError("LLM should not be called when no LLM metric is selected")

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


class _BadContentProvider:
    def __init__(self, content: str) -> None:
        self._content = content

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        return LLMResponse(content=self._content, usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


# --- image -------------------------------------------------------------------

_IMAGE_CLS_ALL = frozenset(
    {"label_balance", "prompt_label_consistency", "prompt_diversity", "duplicate_rate"}
)


def _image_cls_frame() -> pd.DataFrame:
    # prompts: 3 unique / 4 rows -> prompt_diversity 0.75, duplicate_rate 0.25.
    # labels: balanced 2/2 -> label_balance (normalized entropy) = 1.0.
    return pd.DataFrame(
        {
            "prompt": ["cat on mat", "cat on mat", "dog in park", "dog in yard"],
            "label": ["cat", "cat", "dog", "dog"],
            "object_key": ["img/0.png", "img/1.png", "img/2.png", "img/3.png"],
            "mime_type": ["image/png"] * 4,
        }
    )


async def test_image_classification_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    llm = _FixedFakeProvider({"consistent": [True, True, False, True]})  # 3/4 -> 0.75
    ctx = EvaluationContext(
        subject=_image_cls_frame(), task_type=TaskType.IMAGE_CLASSIFICATION, sample_rows=4
    )
    prov = provider_for(TaskType.IMAGE_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_IMAGE_CLS_ALL)  # type: ignore[arg-type]
    assert score.dimension.value == "task_quality"
    assert score.metrics["label_balance"] == pytest.approx(1.0)
    assert score.metrics["prompt_diversity"] == pytest.approx(0.75)
    assert score.metrics["duplicate_rate"] == pytest.approx(0.25)
    assert score.metrics["prompt_label_consistency"] == pytest.approx(0.75)
    expected_score = sum(score.metrics[k] for k in _IMAGE_CLS_ALL) / len(_IMAGE_CLS_ALL)
    assert score.score == pytest.approx(expected_score)
    assert llm.calls[0].params.get("temperature") == 0


async def test_image_classification_provider_catalog() -> None:
    prov = provider_for(TaskType.IMAGE_CLASSIFICATION)
    assert prov is not None
    keys = {m.key for m in prov.metric_catalog()}
    assert keys == _IMAGE_CLS_ALL
    llm_keys = {m.key for m in prov.metric_catalog() if m.requires_llm}
    assert llm_keys == {"prompt_label_consistency"}


async def test_image_classification_provider_skips_llm_when_not_selected(
    model_cfg: ModelConfig,
) -> None:
    ctx = EvaluationContext(
        subject=_image_cls_frame(), task_type=TaskType.IMAGE_CLASSIFICATION, sample_rows=4
    )
    prov = provider_for(TaskType.IMAGE_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"label_balance", "prompt_diversity", "duplicate_rate"}),
    )
    assert set(score.metrics) == {"label_balance", "prompt_diversity", "duplicate_rate"}


async def test_image_classification_provider_missing_label_column_raises(
    model_cfg: ModelConfig,
) -> None:
    df = _image_cls_frame().drop(columns=["label"])
    ctx = EvaluationContext(subject=df, task_type=TaskType.IMAGE_CLASSIFICATION, sample_rows=4)
    prov = provider_for(TaskType.IMAGE_CLASSIFICATION)
    assert prov is not None
    # label_balance not selected -> fine, no column requirement triggered.
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"prompt_diversity"}),
    )
    assert score.metrics["prompt_diversity"] == pytest.approx(0.75)
    # label_balance selected -> missing column raises.
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _ExplodingProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"label_balance"}),
        )


async def test_image_generation_provider_has_no_label_metrics_and_works_without_label(
    model_cfg: ModelConfig,
) -> None:
    df = pd.DataFrame(
        {
            "prompt": ["a red car", "a red car", "a blue bike"],
            "object_key": ["img/0.png", "img/1.png", "img/2.png"],
            "mime_type": ["image/png"] * 3,
        }
    )
    ctx = EvaluationContext(subject=df, task_type=TaskType.IMAGE_GENERATION, sample_rows=3)
    prov = provider_for(TaskType.IMAGE_GENERATION)
    assert prov is not None
    keys = {m.key for m in prov.metric_catalog()}
    assert keys == {"prompt_diversity", "duplicate_rate"}
    assert not any(m.requires_llm for m in prov.metric_catalog())
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"prompt_diversity", "duplicate_rate"}),
    )
    assert score.metrics["prompt_diversity"] == pytest.approx(2 / 3)
    assert score.metrics["duplicate_rate"] == pytest.approx(1 / 3)


# --- audio -------------------------------------------------------------------

_AUDIO_CLS_ALL = frozenset(
    {"label_balance", "transcript_label_consistency", "duration_uniformity", "transcript_quality"}
)


def _audio_cls_frame() -> pd.DataFrame:
    # labels: balanced 2/2 -> label_balance = 1.0.
    # duration_seconds: median 5.0 (sorted [5,5,5,12], median of middle two = 5.0);
    # +/-10% band = [4.5, 5.5] -> 3 of 4 rows in-band -> duration_uniformity 0.75.
    return pd.DataFrame(
        {
            "text": ["yes I agree", "yes indeed", "no thanks", "no way"],
            "label": ["yes", "yes", "no", "no"],
            "duration_seconds": [5.0, 5.0, 5.0, 12.0],
        }
    )


async def test_audio_classification_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    # Single fake payload carries both the "consistent" array (for
    # transcript_label_consistency) and "transcript_quality" (for the rubric) -- both
    # LLM metrics are folded into ONE combined call/response (see media.py's
    # `_AudioBase._oracle`), so this single payload satisfies both parses.
    llm = _FixedFakeProvider({"consistent": [True, False, True, True], "transcript_quality": 4})
    ctx = EvaluationContext(
        subject=_audio_cls_frame(), task_type=TaskType.AUDIO_CLASSIFICATION, sample_rows=4
    )
    prov = provider_for(TaskType.AUDIO_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_AUDIO_CLS_ALL)  # type: ignore[arg-type]
    assert score.metrics["label_balance"] == pytest.approx(1.0)
    assert score.metrics["duration_uniformity"] == pytest.approx(0.75)
    assert score.metrics["transcript_label_consistency"] == pytest.approx(0.75)
    assert score.metrics["transcript_quality"] == pytest.approx(0.8)
    expected_score = sum(score.metrics[k] for k in _AUDIO_CLS_ALL) / len(_AUDIO_CLS_ALL)
    assert score.score == pytest.approx(expected_score)
    assert len(llm.calls) == 1  # both LLM metrics are folded into a single call
    assert all(c.params.get("temperature") == 0 for c in llm.calls)


async def test_audio_classification_provider_single_metric_makes_one_call(
    model_cfg: ModelConfig,
) -> None:
    # Only transcript_label_consistency selected -> exactly one call, and the payload
    # need not carry "transcript_quality" at all (lenient: only the requested key is
    # parsed).
    llm = _FixedFakeProvider({"consistent": [True, False, True, True]})
    ctx = EvaluationContext(
        subject=_audio_cls_frame(), task_type=TaskType.AUDIO_CLASSIFICATION, sample_rows=4
    )
    prov = provider_for(TaskType.AUDIO_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(
        ctx,
        llm,  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"transcript_label_consistency"}),
    )
    assert score.metrics["transcript_label_consistency"] == pytest.approx(0.75)
    assert len(llm.calls) == 1

    # Only transcript_quality selected -> exactly one call, payload need not carry
    # "consistent".
    llm2 = _FixedFakeProvider({"transcript_quality": 4})
    score2 = await prov.score(
        ctx,
        llm2,  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"transcript_quality"}),
    )
    assert score2.metrics["transcript_quality"] == pytest.approx(0.8)
    assert len(llm2.calls) == 1


async def test_audio_classification_provider_catalog() -> None:
    prov = provider_for(TaskType.AUDIO_CLASSIFICATION)
    assert prov is not None
    keys = {m.key for m in prov.metric_catalog()}
    assert keys == _AUDIO_CLS_ALL
    llm_keys = {m.key for m in prov.metric_catalog() if m.requires_llm}
    assert llm_keys == {"transcript_label_consistency", "transcript_quality"}


async def test_audio_classification_provider_skips_llm_when_not_selected(
    model_cfg: ModelConfig,
) -> None:
    ctx = EvaluationContext(
        subject=_audio_cls_frame(), task_type=TaskType.AUDIO_CLASSIFICATION, sample_rows=4
    )
    prov = provider_for(TaskType.AUDIO_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"label_balance", "duration_uniformity"}),
    )
    assert set(score.metrics) == {"label_balance", "duration_uniformity"}


async def test_audio_classification_provider_missing_duration_column_raises(
    model_cfg: ModelConfig,
) -> None:
    df = _audio_cls_frame().drop(columns=["duration_seconds"])
    ctx = EvaluationContext(subject=df, task_type=TaskType.AUDIO_CLASSIFICATION, sample_rows=4)
    prov = provider_for(TaskType.AUDIO_CLASSIFICATION)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _ExplodingProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"duration_uniformity"}),
        )


async def test_speech_synthesis_provider_has_no_label_metrics_and_works_without_label(
    model_cfg: ModelConfig,
) -> None:
    df = pd.DataFrame(
        {
            "text": ["hello there", "good morning", "good night"],
            "duration_seconds": [4.0, 4.2, 8.0],
            "voice": ["v1", "v1", "v2"],
        }
    )
    llm = _FixedFakeProvider({"transcript_quality": 5})
    ctx = EvaluationContext(subject=df, task_type=TaskType.SPEECH_SYNTHESIS, sample_rows=3)
    prov = provider_for(TaskType.SPEECH_SYNTHESIS)
    assert prov is not None
    keys = {m.key for m in prov.metric_catalog()}
    assert keys == {"duration_uniformity", "transcript_quality"}
    score = await prov.score(
        ctx,
        llm,  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"duration_uniformity", "transcript_quality"}),
    )
    # median 4.2, +/-10% band [3.78, 4.62] -> rows 4.0 and 4.2 in-band, 8.0 out -> 2/3.
    assert score.metrics["duration_uniformity"] == pytest.approx(2 / 3)
    assert score.metrics["transcript_quality"] == pytest.approx(1.0)


async def test_audio_consistency_oracle_parse_errors(model_cfg: ModelConfig) -> None:
    ctx = EvaluationContext(
        subject=_audio_cls_frame(), task_type=TaskType.AUDIO_CLASSIFICATION, sample_rows=4
    )
    prov = provider_for(TaskType.AUDIO_CLASSIFICATION)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _BadContentProvider("not json"),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"transcript_label_consistency"}),
        )
    mismatched = json.dumps({"consistent": [True, False]})
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _BadContentProvider(mismatched),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"transcript_label_consistency"}),
        )
    bad_rubric = json.dumps({"transcript_quality": 7})  # out of 1-5 range
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _BadContentProvider(bad_rubric),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"transcript_quality"}),
        )


# --- video -------------------------------------------------------------------

_VIDEO_ALL = frozenset(
    {
        "duration_conformance",
        "resolution_consistency",
        "fps_consistency",
        "prompt_diversity",
        "prompt_quality",
    }
)


def _video_frame() -> pd.DataFrame:
    # duration_seconds: sorted [10,10,10,10,20], median = 10; +/-10% band [9, 11] ->
    # 4 of 5 rows in-band -> duration_conformance 0.8.
    # (width, height): modal (1920, 1080) appears in 4 of 5 rows -> 0.8.
    # fps: modal 24 appears in 4 of 5 rows -> 0.8.
    # prompt: 4 unique of 5 rows ("p2" repeats) -> prompt_diversity 0.8.
    return pd.DataFrame(
        {
            "prompt": ["p1", "p2", "p3", "p2", "p5"],
            "duration_seconds": [10.0, 10.0, 10.0, 20.0, 10.0],
            "width": [1920, 1920, 1920, 1280, 1920],
            "height": [1080, 1080, 1080, 720, 1080],
            "fps": [24, 24, 30, 24, 24],
        }
    )


async def test_video_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    llm = _FixedFakeProvider({"prompt_quality": 4})  # 4/5 -> 0.8
    ctx = EvaluationContext(subject=_video_frame(), task_type=TaskType.TEXT_TO_VIDEO, sample_rows=5)
    prov = provider_for(TaskType.TEXT_TO_VIDEO)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_VIDEO_ALL)  # type: ignore[arg-type]
    assert score.metrics["duration_conformance"] == pytest.approx(0.8)
    assert score.metrics["resolution_consistency"] == pytest.approx(0.8)
    assert score.metrics["fps_consistency"] == pytest.approx(0.8)
    assert score.metrics["prompt_diversity"] == pytest.approx(0.8)
    assert score.metrics["prompt_quality"] == pytest.approx(0.8)
    expected_score = sum(score.metrics[k] for k in _VIDEO_ALL) / len(_VIDEO_ALL)
    assert score.score == pytest.approx(expected_score)


async def test_video_provider_catalog() -> None:
    prov = provider_for(TaskType.TEXT_TO_VIDEO)
    assert prov is not None
    keys = {m.key for m in prov.metric_catalog()}
    assert keys == _VIDEO_ALL
    llm_keys = {m.key for m in prov.metric_catalog() if m.requires_llm}
    assert llm_keys == {"prompt_quality"}


async def test_video_provider_skips_llm_when_not_selected(model_cfg: ModelConfig) -> None:
    ctx = EvaluationContext(subject=_video_frame(), task_type=TaskType.TEXT_TO_VIDEO, sample_rows=5)
    prov = provider_for(TaskType.TEXT_TO_VIDEO)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset(
            {
                "duration_conformance",
                "resolution_consistency",
                "fps_consistency",
                "prompt_diversity",
            }
        ),
    )
    assert set(score.metrics) == {
        "duration_conformance",
        "resolution_consistency",
        "fps_consistency",
        "prompt_diversity",
    }


async def test_video_provider_resolution_consistency_deterministic_tie_break(
    model_cfg: ModelConfig,
) -> None:
    # Two distinct (width, height) pairs each appear twice -> a count tie. The
    # deterministic tie-break picks the smallest pair by tuple sort order --
    # (1280, 720) < (1920, 1080) -- so resolution_consistency is the fraction of rows
    # equal to (1280, 720): 2/4 = 0.5. This must not depend on pandas' internal
    # `value_counts()` tie order.
    df = pd.DataFrame(
        {
            "prompt": ["p1", "p2", "p3", "p4"],
            "width": [1920, 1920, 1280, 1280],
            "height": [1080, 1080, 720, 720],
        }
    )
    ctx = EvaluationContext(subject=df, task_type=TaskType.TEXT_TO_VIDEO, sample_rows=4)
    prov = provider_for(TaskType.TEXT_TO_VIDEO)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"resolution_consistency"}),
    )
    assert score.metrics["resolution_consistency"] == pytest.approx(0.5)


async def test_video_provider_missing_resolution_columns_raises(model_cfg: ModelConfig) -> None:
    df = _video_frame().drop(columns=["width", "height"])
    ctx = EvaluationContext(subject=df, task_type=TaskType.TEXT_TO_VIDEO, sample_rows=5)
    prov = provider_for(TaskType.TEXT_TO_VIDEO)
    assert prov is not None
    # Unselected -> fine.
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"fps_consistency"}),
    )
    assert score.metrics["fps_consistency"] == pytest.approx(0.8)
    # Selected -> raises.
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _ExplodingProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"resolution_consistency"}),
        )
