"""Fidelity expert: statistical distribution similarity synthetic-vs-reference.

Mirrors the *kind* of drift metrics Evidently reports (per-column drift +
correlation structure) but computed directly on scipy: KS statistic for numeric
columns, Jensen-Shannon distance for categoricals, and a correlation-matrix
delta. No LLM. Requires a reference dataset (else `JudgeNotApplicable`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from scipy.spatial.distance import jensenshannon  # type: ignore[import-untyped]
from scipy.stats import ks_2samp  # type: ignore[import-untyped]

from anodyne_evaluation.judges.base import (
    StatisticalJudge,
    categorical_columns,
    clamp01,
    numeric_columns,
    shared_columns,
)
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


def _js_distance(a: pd.Series, b: pd.Series) -> float:
    """Jensen-Shannon distance between two categorical value distributions.

    Frequencies are aligned over the union of categories so both probability
    vectors share support (required for a well-defined divergence).
    """
    pa = a.astype(str).value_counts(normalize=True)
    pb = b.astype(str).value_counts(normalize=True)
    cats = sorted(set(pa.index) | set(pb.index))
    va = np.array([pa.get(c, 0.0) for c in cats])
    vb = np.array([pb.get(c, 0.0) for c in cats])
    d = float(jensenshannon(va, vb, base=2))
    return 0.0 if not np.isfinite(d) else d


class FidelityJudge(StatisticalJudge):
    dimension = EvalDimension.FIDELITY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        if ctx.reference is None:
            raise JudgeNotApplicable("fidelity requires a reference dataset")
        syn, ref = ctx.subject, ctx.reference
        cols = shared_columns(syn, ref)
        num = [c for c in numeric_columns(syn) if c in cols]
        cat = [c for c in categorical_columns(syn) if c in cols and c not in num]

        ks = [
            float(ks_2samp(syn[c].dropna(), ref[c].dropna()).statistic)
            for c in num
            if len(syn[c].dropna()) and len(ref[c].dropna())
        ]
        js = [
            _js_distance(syn[c].dropna(), ref[c].dropna())
            for c in cat
            if len(syn[c].dropna()) and len(ref[c].dropna())
        ]

        ks_mean = float(np.mean(ks)) if ks else 0.0
        js_mean = float(np.mean(js)) if js else 0.0
        corr_delta = self._corr_delta(syn, ref, num)

        drift = float(np.mean([ks_mean, js_mean, corr_delta]))
        score = clamp01(1.0 - drift)

        recs: list[str] = []
        if ks_mean > 0.2:
            recs.append("Numeric columns drift from the reference; revisit marginal distributions.")
        if js_mean > 0.2:
            recs.append(
                "Categorical frequencies diverge; check class balance against the reference."
            )
        if corr_delta > 0.2:
            recs.append("Inter-column correlations differ; a copula/joint model may fit better.")
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=(
                f"Distribution similarity to reference: KS={ks_mean:.3f}, "
                f"JS={js_mean:.3f}, correlation-delta={corr_delta:.3f} "
                f"over {len(num)} numeric / {len(cat)} categorical columns."
            ),
            metrics={
                "ks_mean": ks_mean,
                "js_mean": js_mean,
                "correlation_delta": corr_delta,
                "numeric_columns": float(len(num)),
                "categorical_columns": float(len(cat)),
            },
            recommendations=recs,
        )

    @staticmethod
    def _corr_delta(syn: pd.DataFrame, ref: pd.DataFrame, num: list[str]) -> float:
        if len(num) < 2:
            return 0.0
        cs = syn[num].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
        cr = ref[num].apply(pd.to_numeric, errors="coerce").corr().to_numpy()
        diff = np.abs(cs - cr)
        # correlations range over [-1, 1] so an abs difference is in [0, 2];
        # halve to normalize into [0, 1].
        return clamp01(float(np.nanmean(diff)) / 2.0)
