"""`TaskMetricProvider` for `TaskType.TEXT_CLASSIFICATION`.

The exemplar for providers with real metric math (the QA/summarization/tabular/
media providers added in later tasks copy this shape). Two metrics are intrinsic
-- computed straight off the subject frame, no model call needed -- and two are
an LLM-oracle: a single `complete` call asks the model to predict a label for
every sampled text, and accuracy/macro-F1 are derived from comparing those
predictions to the stored labels.
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

_ORACLE_SYSTEM = (
    "You are labeling short texts for a text-classification task. For each numbered "
    "text below, predict the single best label from the candidate label set. Return "
    'ONLY JSON: {"labels": [<label for text 1>, <label for text 2>, ...]} -- an array '
    "with exactly one predicted label per text, in the same order as the texts below. "
    "No prose outside the JSON."
)


class TextClassificationProvider:
    """Standard metrics for the `text_classification` task class."""

    task_type = TaskType.TEXT_CLASSIFICATION

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="accuracy",
                label="Accuracy",
                description="Fraction of sampled texts an LLM oracle labels correctly.",
                requires_llm=True,
            ),
            MetricSpec(
                key="macro_f1",
                label="Macro F1",
                description="Unweighted mean per-class F1 of the LLM oracle's predictions.",
                requires_llm=True,
            ),
            MetricSpec(
                key="class_balance",
                label="Class balance",
                description="Normalized Shannon entropy of the label distribution.",
                requires_llm=False,
            ),
            MetricSpec(
                key="duplicate_rate",
                label="Duplicate rate",
                description="Share of text values that are exact duplicates.",
                requires_llm=False,
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
        if "text" not in df.columns or "label" not in df.columns:
            raise TaskMetricError(
                "text_classification requires 'text' and 'label' columns in the subject frame"
            )

        metrics: dict[str, float] = {}
        if "class_balance" in selected:
            metrics["class_balance"] = self._class_balance(df)
        if "duplicate_rate" in selected:
            metrics["duplicate_rate"] = self._duplicate_rate(df)
        if "accuracy" in selected or "macro_f1" in selected:
            accuracy, macro_f1 = await self._llm_oracle(ctx, provider, model_config)
            if "accuracy" in selected:
                metrics["accuracy"] = accuracy
            if "macro_f1" in selected:
                metrics["macro_f1"] = macro_f1

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("accuracy", 1.0) < 0.7:
            recs.append(
                "LLM-oracle accuracy on the sampled texts is below 0.7; labels may be "
                "noisy or the classes may be hard to distinguish from the text alone."
            )
        if metrics.get("class_balance", 1.0) < 0.5:
            recs.append(
                "The label distribution is skewed (low class balance); consider "
                "rebalancing classes before training on this data."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=(
                f"Text-classification standard metrics for task class '{self.task_type.value}'."
            ),
            metrics=metrics,
            recommendations=recs,
        )

    @staticmethod
    def _class_balance(df: pd.DataFrame) -> float:
        return normalized_label_entropy(df["label"])

    @staticmethod
    def _duplicate_rate(df: pd.DataFrame) -> float:
        n = len(df)
        if n == 0:
            return 0.0
        unique = df["text"].nunique()
        return float(1.0 - unique / n)

    async def _llm_oracle(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
    ) -> tuple[float, float]:
        sample = sample_frame(ctx)
        texts = sample["text"].astype(str).tolist()
        true_labels = sample["label"].astype(str).tolist()
        if not texts:
            return 0.0, 0.0
        candidate_labels = sorted(set(true_labels))
        lines = [f"{i}. {t}" for i, t in enumerate(texts, start=1)]
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_ORACLE_SYSTEM),
                Message(
                    role="user",
                    content=f"Candidate labels: {candidate_labels}\n\nTexts:\n" + "\n".join(lines),
                ),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields reproducible labels.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        predicted = self._parse_labels(resp.content, expected=len(texts))
        accuracy = sum(1 for p, t in zip(predicted, true_labels, strict=True) if p == t) / len(
            texts
        )
        macro_f1 = self._macro_f1(true_labels, predicted, candidate_labels)
        return accuracy, macro_f1

    @staticmethod
    def _parse_labels(raw: str, *, expected: int) -> list[str]:
        text = strip_json(raw)
        try:
            data = json.loads(text)
            labels = data["labels"]
            if not isinstance(labels, list) or len(labels) != expected:
                raise ValueError(f"expected {expected} labels, got {labels!r}")
            return [str(x) for x in labels]
        except Exception as exc:  # json/validation errors -> domain error
            raise TaskMetricError(
                f"could not parse predicted labels from model output: {exc}"
            ) from exc

    @staticmethod
    def _macro_f1(true_labels: list[str], predicted: list[str], classes: list[str]) -> float:
        if not classes:
            return 0.0
        f1s: list[float] = []
        for c in classes:
            tp = sum(1 for t, p in zip(true_labels, predicted, strict=True) if t == c and p == c)
            predicted_c = sum(1 for p in predicted if p == c)
            true_c = sum(1 for t in true_labels if t == c)
            precision = tp / predicted_c if predicted_c else 0.0
            recall = tp / true_c if true_c else 0.0
            f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
            f1s.append(f1)
        return sum(f1s) / len(f1s)


register_provider(TextClassificationProvider())
