"""Diversity / coverage expert: uniqueness, entropy, and mode-collapse.

Subject-only (no reference, no LLM). Flags the classic generative-model failure
where output collapses onto a few modes. Higher score == more diverse.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from anodyne_evaluation.judges.base import (
    StatisticalJudge,
    categorical_columns,
    clamp01,
    numeric_columns,
)
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


def _normalized_entropy(series: pd.Series) -> float:
    """Shannon entropy of a categorical column, normalized by log(k) to [0, 1]."""
    p = series.astype(str).value_counts(normalize=True).to_numpy()
    k = len(p)
    if k <= 1:
        return 0.0
    ent = -np.sum(p * np.log(p))
    return clamp01(float(ent / np.log(k)))


class DiversityJudge(StatisticalJudge):
    dimension = EvalDimension.DIVERSITY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        df = ctx.subject
        n = len(df)
        if n == 0:
            raise JudgeNotApplicable("diversity requires a non-empty dataset")

        uniqueness = [float(df[c].nunique()) / n for c in df.columns if n]
        mean_uniqueness = float(np.mean(uniqueness)) if uniqueness else 0.0

        cats = categorical_columns(df)
        entropies = [_normalized_entropy(df[c].dropna()) for c in cats if len(df[c].dropna())]
        mean_entropy = float(np.mean(entropies)) if entropies else mean_uniqueness

        # Mode collapse: the largest single-value frequency across categorical
        # columns (numeric columns fall back to their top-value frequency too).
        probe = cats or [c for c in numeric_columns(df)]
        mode_freqs = [
            float(df[c].value_counts(normalize=True).iloc[0]) for c in probe if len(df[c].dropna())
        ]
        max_mode_freq = float(np.max(mode_freqs)) if mode_freqs else 0.0

        base = float(np.mean([mean_uniqueness, mean_entropy]))
        # Penalize only severe collapse (a dominant value above 60% frequency).
        penalty = max(0.0, max_mode_freq - 0.6) / 0.4
        score = clamp01(base * (1.0 - penalty))

        recs: list[str] = []
        if max_mode_freq > 0.8:
            recs.append(
                "Severe mode collapse: one value dominates a column; "
                "increase sampling temperature/variety."
            )
        if mean_entropy < 0.4:
            recs.append(
                "Low categorical entropy; broaden the value space for underrepresented fields."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=(
                f"Coverage: mean uniqueness={mean_uniqueness:.3f}, "
                f"mean entropy={mean_entropy:.3f}, "
                f"largest mode frequency={max_mode_freq:.3f}."
            ),
            metrics={
                "mean_uniqueness": mean_uniqueness,
                "mean_entropy": mean_entropy,
                "max_mode_freq": max_mode_freq,
            },
            recommendations=recs,
        )
