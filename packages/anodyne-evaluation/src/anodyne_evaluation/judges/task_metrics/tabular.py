"""`TaskMetricProvider`s for `TaskType.TABULAR_CLASSIFICATION` and
`TaskType.REGRESSION`.

Both providers key off `ctx.target_field` (required -- missing/absent raises
`TaskMetricError`) and treat every other column in the subject frame as a
feature. `TabularClassificationProvider` has one LLM-oracle metric
(`label_consistency`: does the target label look plausible given the row's
feature values) plus two intrinsic ones; `RegressionProvider` is entirely
intrinsic (no LLM call is ever made for it).
"""

from __future__ import annotations

import json
import math

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
    "You are checking whether the target label of tabular rows is consistent with "
    "their feature values. For each numbered row below, decide whether the target "
    "label is a plausible, consistent value given the listed feature values. Return "
    'ONLY JSON: {"consistent": [<bool for row 1>, ...]} -- an array with exactly one '
    "boolean per row, in the same order as the rows below. No prose outside the JSON."
)


def _require_target(ctx: EvaluationContext, task_label: str) -> str:
    target = ctx.target_field
    if not target or target not in ctx.subject.columns:
        raise TaskMetricError(
            f"{task_label} requires 'ctx.target_field' to be set and present in the subject frame"
        )
    return target


def _feature_completeness(df: pd.DataFrame, target_field: str) -> float:
    feature_cols = [c for c in df.columns if c != target_field]
    if not feature_cols or len(df) == 0:
        return 1.0
    null_rates = [df[c].isna().mean() for c in feature_cols]
    return float(1.0 - sum(null_rates) / len(null_rates))


class TabularClassificationProvider:
    """Standard metrics for the `tabular_classification` task class."""

    task_type = TaskType.TABULAR_CLASSIFICATION

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="label_consistency",
                label="Label consistency",
                description=(
                    "Fraction of sampled rows an LLM oracle judges the target label "
                    "consistent with its feature values."
                ),
                requires_llm=True,
            ),
            MetricSpec(
                key="class_balance",
                label="Class balance",
                description="Normalized Shannon entropy of the target label distribution.",
                requires_llm=False,
            ),
            MetricSpec(
                key="feature_completeness",
                label="Feature completeness",
                description="1 minus the mean null-rate across the non-target columns.",
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
        target = _require_target(ctx, "tabular_classification")
        df = ctx.subject

        metrics: dict[str, float] = {}
        if "class_balance" in selected:
            metrics["class_balance"] = self._class_balance(df, target)
        if "feature_completeness" in selected:
            metrics["feature_completeness"] = _feature_completeness(df, target)
        if "label_consistency" in selected:
            metrics["label_consistency"] = await self._llm_oracle(
                ctx, provider, model_config, target
            )

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("label_consistency", 1.0) < 0.7:
            recs.append(
                "LLM-oracle label consistency on the sampled rows is below 0.7; the "
                "target label may not follow plausibly from the feature values."
            )
        if metrics.get("class_balance", 1.0) < 0.5:
            recs.append(
                "The target label distribution is skewed (low class balance); "
                "consider rebalancing classes before training on this data."
            )
        if metrics.get("feature_completeness", 1.0) < 0.9:
            recs.append(
                "Feature completeness is below 0.9; several feature columns have missing values."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=(
                f"Tabular-classification standard metrics for task class '{self.task_type.value}'."
            ),
            metrics=metrics,
            recommendations=recs,
        )

    @staticmethod
    def _class_balance(df: pd.DataFrame, target_field: str) -> float:
        return normalized_label_entropy(df[target_field])

    async def _llm_oracle(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        target_field: str,
    ) -> float:
        sample = sample_frame(ctx)
        if sample.empty:
            return 0.0
        feature_cols = [c for c in sample.columns if c != target_field]
        lines: list[str] = []
        for i, (_, row) in enumerate(sample.iterrows(), start=1):
            features = ", ".join(f"{c}: {row[c]}" for c in feature_cols)
            lines.append(f"{i}. Target: {row[target_field]} | Features: {features}")
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_ORACLE_SYSTEM),
                Message(role="user", content="\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields reproducible judgments.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        consistent = self._parse_consistent(resp.content, expected=len(sample))
        return sum(consistent) / len(consistent)

    @staticmethod
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
                f"could not parse label-consistency judgments from model output: {exc}"
            ) from exc


class RegressionProvider:
    """Standard metrics for the `regression` task class. Entirely intrinsic --
    no metric here needs the LLM, so `score` never calls `provider.complete`."""

    task_type = TaskType.REGRESSION

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="target_range_validity",
                label="Target range validity",
                description="Fraction of target values that are non-null and finite.",
                requires_llm=False,
            ),
            MetricSpec(
                key="target_distribution_health",
                label="Target distribution health",
                description=(
                    "1 minus the clamped absolute Fisher skewness of the target "
                    "column (0 skew -> 1.0 health)."
                ),
                requires_llm=False,
            ),
            MetricSpec(
                key="feature_completeness",
                label="Feature completeness",
                description="1 minus the mean null-rate across the non-target columns.",
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
        target = _require_target(ctx, "regression")
        df = ctx.subject

        metrics: dict[str, float] = {}
        if "target_range_validity" in selected:
            metrics["target_range_validity"] = self._target_range_validity(df, target)
        if "target_distribution_health" in selected:
            metrics["target_distribution_health"] = self._target_distribution_health(df, target)
        if "feature_completeness" in selected:
            metrics["feature_completeness"] = _feature_completeness(df, target)

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("target_range_validity", 1.0) < 0.95:
            recs.append(
                "Target range validity is below 0.95; some target values are null, "
                "NaN, or infinite."
            )
        if metrics.get("target_distribution_health", 1.0) < 0.5:
            recs.append(
                "The target distribution is heavily skewed; consider a transform "
                "(e.g. log) before training a regressor on this data."
            )
        if metrics.get("feature_completeness", 1.0) < 0.9:
            recs.append(
                "Feature completeness is below 0.9; several feature columns have missing values."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=f"Regression standard metrics for task class '{self.task_type.value}'.",
            metrics=metrics,
            recommendations=recs,
        )

    @staticmethod
    def _target_range_validity(df: pd.DataFrame, target_field: str) -> float:
        col = df[target_field]
        if len(col) == 0:
            return 0.0

        def _finite(x: object) -> bool:
            if x is None:
                return False
            try:
                if pd.isna(x):
                    return False
            except (TypeError, ValueError):
                pass
            try:
                return math.isfinite(float(x))  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return False

        return sum(1 for x in col if _finite(x)) / len(col)

    @staticmethod
    def _target_distribution_health(df: pd.DataFrame, target_field: str) -> float:
        skew = df[target_field].skew()
        if pd.isna(skew):
            skew = 0.0
        return float(1 - min(1.0, abs(float(skew)) / 10))


register_provider(TabularClassificationProvider())
register_provider(RegressionProvider())
