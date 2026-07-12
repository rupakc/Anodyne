"""Bias / fairness expert: group representation + outcome disparity.

Requires a `sensitive_field` (else `JudgeNotApplicable`; no LLM). Measures how
balanced the sensitive groups are (representation entropy) and, when a
`target_field` is present, the demographic-parity difference and
disparate-impact ratio of the (binarized) positive outcome across groups.
Higher score == fairer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from anodyne_evaluation.judges.base import StatisticalJudge, clamp01
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


def _positive_rate(y: pd.Series) -> pd.Series:
    """Binarize an outcome column into a positive-class indicator.

    Numeric targets: positive == above the global median. Categorical targets:
    positive == the most frequent (majority) class. Returns the per-row 0/1
    indicator aligned to `y`.
    """
    if pd.api.types.is_numeric_dtype(y):
        return (y > y.median()).astype(int)
    top = y.astype(str).value_counts().index[0]
    return (y.astype(str) == top).astype(int)


class BiasJudge(StatisticalJudge):
    dimension = EvalDimension.BIAS

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        field = ctx.sensitive_field
        if not field or field not in ctx.subject.columns:
            raise JudgeNotApplicable("bias requires a sensitive_field present in the dataset")
        df = ctx.subject
        groups = df[field].dropna().astype(str)
        if groups.empty:
            raise JudgeNotApplicable("bias requires a non-empty sensitive_field")

        rep = groups.value_counts(normalize=True)
        k = len(rep)
        representation_entropy = (
            clamp01(float(-np.sum(rep.to_numpy() * np.log(rep.to_numpy())) / np.log(k)))
            if k > 1
            else 0.0
        )

        dp_diff = 0.0
        di_ratio = 1.0
        metrics: dict[str, float] = {
            "groups": float(k),
            "representation_entropy": representation_entropy,
        }
        recs: list[str] = []

        target = ctx.target_field
        if target and target in df.columns:
            pos = _positive_rate(df[target])
            rates = pos.groupby(df[field].astype(str)).mean()
            if len(rates) > 1:
                dp_diff = clamp01(float(rates.max() - rates.min()))
                di_ratio = float(rates.min() / rates.max()) if rates.max() > 0 else 0.0
                metrics["demographic_parity_diff"] = dp_diff
                metrics["disparate_impact_ratio"] = di_ratio
                if dp_diff > 0.2:
                    recs.append(
                        f"Outcome '{target}' differs by {dp_diff:.1%} across '{field}' "
                        "groups; rebalance or reweight."
                    )
                if di_ratio < 0.8:
                    recs.append(
                        "Disparate impact below the 0.8 four-fifths threshold; review group parity."
                    )
            fairness = 1.0 - dp_diff
            score = clamp01(0.5 * representation_entropy + 0.5 * fairness)
        else:
            score = representation_entropy
            recs.append("No target_field provided; only group representation was assessed.")

        if representation_entropy < 0.5:
            recs.append(
                f"Sensitive groups in '{field}' are imbalanced; increase minority representation."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=(
                f"Fairness on '{field}': representation entropy={representation_entropy:.3f}, "
                f"demographic-parity diff={dp_diff:.3f}, disparate-impact ratio={di_ratio:.3f}."
            ),
            metrics=metrics,
            recommendations=recs,
        )
