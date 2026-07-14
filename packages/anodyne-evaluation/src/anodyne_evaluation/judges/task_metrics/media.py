"""`TaskMetricProvider`s for the media task classes: `IMAGE_CLASSIFICATION`/
`IMAGE_GENERATION`, `AUDIO_CLASSIFICATION`/`SPEECH_SYNTHESIS`, and
`TEXT_TO_VIDEO`.

Media artifacts are loaded (in a later task) as a DataFrame over the manifest
`items`, never as raw pixels/waveform/frame bytes -- so every metric here scores
manifest *fields* (`prompt`, `label`, `duration_seconds`, `width`/`height`,
`fps`, ...), not the underlying media. Every LLM-oracle metric is TEXT-ONLY for
the same reason: it judges the `prompt`/`text` manifest field against the
stated `label`, or rates prompt/transcript quality from text alone -- no
image/audio/video perception is ever attempted.

Image and audio each share one implementation base across their two task-type
variants; only `metric_catalog()` (and therefore which columns `score` ends up
requiring) differs between the classification and generation/synthesis
variant, since generation/synthesis manifests carry no `label` column. Video
has a single task type, so it needs no base/subclass split.
"""

from __future__ import annotations

import json

import pandas as pd  # type: ignore[import-untyped]
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.judges.task_metrics.base import (
    TaskMetricError,
    mean_contribution,
    normalized_label_entropy,
    sample_frame,
    strip_json,
)
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, register_provider

_IMAGE_CONSISTENCY_SYSTEM = (
    "You are checking whether an image-generation prompt plausibly describes its "
    "stated label, using only the prompt text (no image is available to you). For "
    "each numbered item below, decide whether the prompt is a plausible description "
    'of the label. Return ONLY JSON: {"consistent": [<bool for item 1>, ...]} -- an '
    "array with exactly one boolean per item, in the same order as the items below. "
    "No prose outside the JSON."
)

_AUDIO_CONSISTENCY_SYSTEM = (
    "You are checking whether a spoken-audio transcript plausibly matches its stated "
    "label, using only the transcript text (no audio is available to you). For each "
    "numbered item below, decide whether the transcript is plausibly consistent with "
    'the label. Return ONLY JSON: {"consistent": [<bool for item 1>, ...]} -- an array '
    "with exactly one boolean per item, in the same order as the items below. No "
    "prose outside the JSON."
)

_AUDIO_QUALITY_SYSTEM = (
    "You are rating a batch of speech-synthesis transcripts for text quality, using "
    "only the transcript text (no audio is available to you). Considering all "
    "transcripts below together as a single batch, rate ONE criterion, an INTEGER "
    "from 1 (poor) to 5 (excellent): transcript_quality (transcripts are well-formed, "
    'natural, and suitable to be read aloud). Return ONLY JSON: {"transcript_quality": '
    "int}. No prose outside the JSON."
)

_VIDEO_QUALITY_SYSTEM = (
    "You are rating a batch of text-to-video generation prompts for prompt quality, "
    "using only the prompt text (no video is available to you). Considering all "
    "prompts below together as a single batch, rate ONE criterion, an INTEGER from 1 "
    "(poor) to 5 (excellent): prompt_quality (prompts are clear, specific, and "
    'well-formed as generation instructions). Return ONLY JSON: {"prompt_quality": '
    "int}. No prose outside the JSON."
)


# --- shared intrinsic helpers ----------------------------------------------


def _require_columns(df: pd.DataFrame, cols: set[str], task_label: str) -> None:
    missing = cols - set(df.columns)
    if missing:
        raise TaskMetricError(
            f"{task_label} requires column(s) {sorted(missing)} for the selected metrics"
        )


def _diversity(df: pd.DataFrame, col: str) -> float:
    """`unique(col) / len(df)` -- shared by `prompt_diversity` (image, video) and
    `duplicate_rate` (image, as `1 - _diversity(...)`)."""
    if len(df) == 0:
        return 0.0
    return float(df[col].nunique() / len(df))


def _median_band_fraction(df: pd.DataFrame, col: str = "duration_seconds") -> float:
    """Fraction of rows within +/-10% of the column's median -- `duration_uniformity`
    (audio) and `duration_conformance` (video) are the same formula on the same
    manifest field, just named differently per modality."""
    if len(df) == 0:
        return 0.0
    median = df[col].median()
    lo, hi = median * 0.9, median * 1.1
    return float(((df[col] >= lo) & (df[col] <= hi)).mean())


def _modal_fraction(series: pd.Series) -> float:
    """Fraction of values equal to the series' mode -- `fps_consistency`."""
    if len(series) == 0:
        return 0.0
    mode = series.mode()
    if mode.empty:
        return 0.0
    return float((series == mode.iloc[0]).mean())


def _resolution_consistency(df: pd.DataFrame) -> float:
    """Fraction of rows whose `(width, height)` pair equals the modal pair."""
    if len(df) == 0:
        return 0.0
    pairs = list(zip(df["width"], df["height"], strict=True))
    counts = pd.Series(pairs).value_counts()
    modal = counts.index[0]
    return sum(1 for p in pairs if p == modal) / len(pairs)


# --- shared LLM-oracle parsing ----------------------------------------------


def _parse_consistent(raw: str, *, expected: int) -> list[bool]:
    text = strip_json(raw)
    try:
        data = json.loads(text)
        vals = data["consistent"]
        if not isinstance(vals, list) or len(vals) != expected:
            raise ValueError(f"expected {expected} entries, got {vals!r}")
        return [bool(x) for x in vals]
    except Exception as exc:  # json/validation errors -> domain error
        raise TaskMetricError(
            f"could not parse consistency judgments from model output: {exc}"
        ) from exc


def _parse_single_rubric(raw: str, key: str) -> float:
    text = strip_json(raw)
    try:
        data = json.loads(text)
        v = data[key]
        if isinstance(v, bool) or not isinstance(v, int) or not (1 <= v <= 5):
            raise ValueError(f"{key} must be an integer 1-5, got {v!r}")
        return v / 5.0
    except Exception as exc:  # json/validation errors -> domain error
        raise TaskMetricError(f"could not parse {key} rubric from model output: {exc}") from exc


# --- image ------------------------------------------------------------------

_IMAGE_CATALOG: dict[str, MetricSpec] = {
    "label_balance": MetricSpec(
        key="label_balance",
        label="Label balance",
        description="Normalized Shannon entropy of the label distribution.",
        requires_llm=False,
    ),
    "prompt_label_consistency": MetricSpec(
        key="prompt_label_consistency",
        label="Prompt/label consistency",
        description=(
            "Fraction of sampled prompts an LLM oracle judges as a plausible "
            "description of the stated label (text-only, no image perception)."
        ),
        requires_llm=True,
    ),
    "prompt_diversity": MetricSpec(
        key="prompt_diversity",
        label="Prompt diversity",
        description="Unique prompts divided by row count.",
        requires_llm=False,
    ),
    "duplicate_rate": MetricSpec(
        key="duplicate_rate",
        label="Duplicate rate",
        description="1 minus unique prompts divided by row count.",
        requires_llm=False,
    ),
}


class _ImageBase:
    """Shared metric logic for `IMAGE_CLASSIFICATION`/`IMAGE_GENERATION`. Subclasses
    fix `task_type` and `_catalog_keys()`; generation manifests carry no `label`
    column, so `ImageGenerationProvider` omits the label-dependent keys."""

    task_type: TaskType

    def _catalog_keys(self) -> tuple[str, ...]:
        raise NotImplementedError

    def metric_catalog(self) -> list[MetricSpec]:
        return [_IMAGE_CATALOG[k] for k in self._catalog_keys()]

    async def score(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        *,
        selected: frozenset[str],
    ) -> ExpertScore:
        df = ctx.subject
        needed: set[str] = set()
        if {"label_balance", "prompt_label_consistency"} & selected:
            needed.add("label")
        if {"prompt_diversity", "duplicate_rate", "prompt_label_consistency"} & selected:
            needed.add("prompt")
        _require_columns(df, needed, f"'{self.task_type.value}'")

        metrics: dict[str, float] = {}
        if "label_balance" in selected:
            metrics["label_balance"] = normalized_label_entropy(df["label"])
        if "prompt_diversity" in selected:
            metrics["prompt_diversity"] = _diversity(df, "prompt")
        if "duplicate_rate" in selected:
            metrics["duplicate_rate"] = 1.0 - _diversity(df, "prompt")
        if "prompt_label_consistency" in selected:
            metrics["prompt_label_consistency"] = await self._consistency_oracle(
                ctx, provider, model_config
            )

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("prompt_label_consistency", 1.0) < 0.7:
            recs.append(
                "LLM-oracle prompt/label consistency is below 0.7; prompts may not "
                "plausibly describe their stated label."
            )
        if metrics.get("label_balance", 1.0) < 0.5:
            recs.append(
                "The label distribution is skewed (low label balance); consider "
                "rebalancing classes before training on this data."
            )
        if metrics.get("duplicate_rate", 0.0) > 0.3:
            recs.append("Duplicate rate is above 0.3; many prompts are exact repeats.")
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=f"Image standard metrics for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    async def _consistency_oracle(
        self, ctx: EvaluationContext, provider: LLMProvider, model_config: ModelConfig
    ) -> float:
        sample = sample_frame(ctx)
        if sample.empty:
            return 0.0
        prompts = sample["prompt"].astype(str).tolist()
        labels = sample["label"].astype(str).tolist()
        lines = [
            f"{i}. Prompt: {p}\nLabel: {lab}"
            for i, (p, lab) in enumerate(zip(prompts, labels, strict=True), start=1)
        ]
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_IMAGE_CONSISTENCY_SYSTEM),
                Message(role="user", content="\n\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields reproducible judgments.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        consistent = _parse_consistent(resp.content, expected=len(prompts))
        return sum(consistent) / len(consistent)


class ImageClassificationProvider(_ImageBase):
    """Standard metrics for the `image_classification` task class."""

    task_type = TaskType.IMAGE_CLASSIFICATION

    def _catalog_keys(self) -> tuple[str, ...]:
        return (
            "label_balance",
            "prompt_label_consistency",
            "prompt_diversity",
            "duplicate_rate",
        )


class ImageGenerationProvider(_ImageBase):
    """Standard metrics for the `image_generation` task class. No label-dependent
    metrics -- image-generation manifests carry no `label` column."""

    task_type = TaskType.IMAGE_GENERATION

    def _catalog_keys(self) -> tuple[str, ...]:
        return ("prompt_diversity", "duplicate_rate")


# --- audio -------------------------------------------------------------------

_AUDIO_CATALOG: dict[str, MetricSpec] = {
    "label_balance": MetricSpec(
        key="label_balance",
        label="Label balance",
        description="Normalized Shannon entropy of the label distribution.",
        requires_llm=False,
    ),
    "transcript_label_consistency": MetricSpec(
        key="transcript_label_consistency",
        label="Transcript/label consistency",
        description=(
            "Fraction of sampled transcripts an LLM oracle judges as plausibly "
            "consistent with the stated label (text-only, no audio perception)."
        ),
        requires_llm=True,
    ),
    "duration_uniformity": MetricSpec(
        key="duration_uniformity",
        label="Duration uniformity",
        description="Fraction of rows within +/-10% of the median duration.",
        requires_llm=False,
    ),
    "transcript_quality": MetricSpec(
        key="transcript_quality",
        label="Transcript quality",
        description="LLM-judged 1-5 rating of transcript text quality.",
        requires_llm=True,
    ),
}


class _AudioBase:
    """Shared metric logic for `AUDIO_CLASSIFICATION`/`SPEECH_SYNTHESIS`. Subclasses
    fix `task_type` and `_catalog_keys()`; synthesis manifests carry no `label`
    column, so `SpeechSynthesisProvider` omits the label-dependent keys."""

    task_type: TaskType

    def _catalog_keys(self) -> tuple[str, ...]:
        raise NotImplementedError

    def metric_catalog(self) -> list[MetricSpec]:
        return [_AUDIO_CATALOG[k] for k in self._catalog_keys()]

    async def score(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        *,
        selected: frozenset[str],
    ) -> ExpertScore:
        df = ctx.subject
        needed: set[str] = set()
        if {"label_balance", "transcript_label_consistency"} & selected:
            needed.add("label")
        if {"transcript_label_consistency", "transcript_quality"} & selected:
            needed.add("text")
        if "duration_uniformity" in selected:
            needed.add("duration_seconds")
        _require_columns(df, needed, f"'{self.task_type.value}'")

        metrics: dict[str, float] = {}
        if "label_balance" in selected:
            metrics["label_balance"] = normalized_label_entropy(df["label"])
        if "duration_uniformity" in selected:
            metrics["duration_uniformity"] = _median_band_fraction(df)
        if "transcript_label_consistency" in selected:
            metrics["transcript_label_consistency"] = await self._consistency_oracle(
                ctx, provider, model_config
            )
        if "transcript_quality" in selected:
            metrics["transcript_quality"] = await self._quality_oracle(ctx, provider, model_config)

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("transcript_label_consistency", 1.0) < 0.7:
            recs.append(
                "LLM-oracle transcript/label consistency is below 0.7; transcripts may "
                "not plausibly match their stated label."
            )
        if metrics.get("transcript_quality", 1.0) < 0.6:
            recs.append("LLM-judged transcript quality is low; review transcript text.")
        if metrics.get("label_balance", 1.0) < 0.5:
            recs.append(
                "The label distribution is skewed (low label balance); consider "
                "rebalancing classes before training on this data."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=f"Audio standard metrics for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    async def _consistency_oracle(
        self, ctx: EvaluationContext, provider: LLMProvider, model_config: ModelConfig
    ) -> float:
        sample = sample_frame(ctx)
        if sample.empty:
            return 0.0
        texts = sample["text"].astype(str).tolist()
        labels = sample["label"].astype(str).tolist()
        lines = [
            f"{i}. Transcript: {t}\nLabel: {lab}"
            for i, (t, lab) in enumerate(zip(texts, labels, strict=True), start=1)
        ]
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_AUDIO_CONSISTENCY_SYSTEM),
                Message(role="user", content="\n\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields reproducible judgments.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        consistent = _parse_consistent(resp.content, expected=len(texts))
        return sum(consistent) / len(consistent)

    async def _quality_oracle(
        self, ctx: EvaluationContext, provider: LLMProvider, model_config: ModelConfig
    ) -> float:
        sample = sample_frame(ctx)
        if sample.empty:
            return 0.0
        texts = sample["text"].astype(str).tolist()
        lines = [f"{i}. {t}" for i, t in enumerate(texts, start=1)]
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_AUDIO_QUALITY_SYSTEM),
                Message(role="user", content="\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields a reproducible verdict.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        return _parse_single_rubric(resp.content, "transcript_quality")


class AudioClassificationProvider(_AudioBase):
    """Standard metrics for the `audio_classification` task class."""

    task_type = TaskType.AUDIO_CLASSIFICATION

    def _catalog_keys(self) -> tuple[str, ...]:
        return (
            "label_balance",
            "transcript_label_consistency",
            "duration_uniformity",
            "transcript_quality",
        )


class SpeechSynthesisProvider(_AudioBase):
    """Standard metrics for the `speech_synthesis` task class. No label-dependent
    metrics -- speech-synthesis manifests carry no `label` column."""

    task_type = TaskType.SPEECH_SYNTHESIS

    def _catalog_keys(self) -> tuple[str, ...]:
        return ("duration_uniformity", "transcript_quality")


# --- video ---------------------------------------------------------------


class VideoTaskProvider:
    """Standard metrics for the `text_to_video` task class. Video manifests carry
    no `label` column at all, so unlike image/audio there is only one task type
    and no classification/generation split."""

    task_type = TaskType.TEXT_TO_VIDEO

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="duration_conformance",
                label="Duration conformance",
                description="Fraction of rows within +/-10% of the median duration.",
                requires_llm=False,
            ),
            MetricSpec(
                key="resolution_consistency",
                label="Resolution consistency",
                description="Fraction of rows whose (width, height) equals the modal pair.",
                requires_llm=False,
            ),
            MetricSpec(
                key="fps_consistency",
                label="FPS consistency",
                description="Fraction of rows equal to the modal fps value.",
                requires_llm=False,
            ),
            MetricSpec(
                key="prompt_diversity",
                label="Prompt diversity",
                description="Unique prompts divided by row count.",
                requires_llm=False,
            ),
            MetricSpec(
                key="prompt_quality",
                label="Prompt quality",
                description=(
                    "LLM-judged 1-5 rating of generation-prompt quality (text-only, "
                    "no video perception)."
                ),
                requires_llm=True,
            ),
        ]

    async def score(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        *,
        selected: frozenset[str],
    ) -> ExpertScore:
        df = ctx.subject
        needed: set[str] = set()
        if "duration_conformance" in selected:
            needed.add("duration_seconds")
        if "resolution_consistency" in selected:
            needed |= {"width", "height"}
        if "fps_consistency" in selected:
            needed.add("fps")
        if {"prompt_diversity", "prompt_quality"} & selected:
            needed.add("prompt")
        _require_columns(df, needed, f"'{self.task_type.value}'")

        metrics: dict[str, float] = {}
        if "duration_conformance" in selected:
            metrics["duration_conformance"] = _median_band_fraction(df)
        if "resolution_consistency" in selected:
            metrics["resolution_consistency"] = _resolution_consistency(df)
        if "fps_consistency" in selected:
            metrics["fps_consistency"] = _modal_fraction(df["fps"])
        if "prompt_diversity" in selected:
            metrics["prompt_diversity"] = _diversity(df, "prompt")
        if "prompt_quality" in selected:
            metrics["prompt_quality"] = await self._quality_oracle(ctx, provider, model_config)

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("prompt_quality", 1.0) < 0.6:
            recs.append("LLM-judged prompt quality is low; review generation prompts.")
        if metrics.get("resolution_consistency", 1.0) < 0.9:
            recs.append("Resolution consistency is below 0.9; clips vary in (width, height).")
        if metrics.get("duration_conformance", 1.0) < 0.9:
            recs.append(
                "Duration conformance is below 0.9; clip durations vary widely around the median."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=f"Video standard metrics for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    async def _quality_oracle(
        self, ctx: EvaluationContext, provider: LLMProvider, model_config: ModelConfig
    ) -> float:
        sample = sample_frame(ctx)
        if sample.empty:
            return 0.0
        prompts = sample["prompt"].astype(str).tolist()
        lines = [f"{i}. {p}" for i, p in enumerate(prompts, start=1)]
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_VIDEO_QUALITY_SYSTEM),
                Message(role="user", content="\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields a reproducible verdict.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        return _parse_single_rubric(resp.content, "prompt_quality")


register_provider(ImageClassificationProvider())
register_provider(ImageGenerationProvider())
register_provider(AudioClassificationProvider())
register_provider(SpeechSynthesisProvider())
register_provider(VideoTaskProvider())
