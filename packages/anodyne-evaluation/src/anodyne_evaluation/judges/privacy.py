"""Privacy / leakage expert: memorization + nearest-neighbour distance ratio.

Requires a reference dataset (else `JudgeNotApplicable`; no LLM). Two risks:
- **Exact duplicates**: synthetic rows reproduced verbatim from the reference
  (outright memorization).
- **Distance to closest record (DCR)**: on standardized numeric columns, how
  close synthetic rows sit to their nearest real neighbour; near-zero distance
  means the generator is effectively copying real individuals.
Higher score == lower leakage risk.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.neighbors import NearestNeighbors  # type: ignore[import-untyped]

from anodyne_evaluation.judges.base import (
    StatisticalJudge,
    clamp01,
    numeric_columns,
    shared_columns,
)
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


class PrivacyJudge(StatisticalJudge):
    dimension = EvalDimension.PRIVACY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        if ctx.reference is None:
            raise JudgeNotApplicable("privacy requires a reference dataset")
        syn, ref = ctx.subject, ctx.reference
        cols = shared_columns(syn, ref)
        if not cols or len(syn) == 0:
            raise JudgeNotApplicable("privacy requires overlapping, non-empty columns")

        exact_dup_rate = self._exact_duplicate_rate(syn[cols], ref[cols])
        mean_nn, dcr_risk = self._dcr(syn, ref, [c for c in numeric_columns(syn) if c in cols])

        # Memorization dominates the risk; DCR proximity is secondary.
        risk = clamp01(max(exact_dup_rate, 0.5 * exact_dup_rate + 0.5 * dcr_risk))
        score = clamp01(1.0 - risk)

        recs: list[str] = []
        if exact_dup_rate > 0.01:
            recs.append(
                f"{exact_dup_rate:.1%} of rows are verbatim copies of reference records; "
                "add differential noise or dedup."
            )
        if dcr_risk > 0.5:
            recs.append(
                "Synthetic rows sit very close to real records (low DCR); increase generalization."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=(
                f"Leakage: exact-duplicate rate={exact_dup_rate:.3f}, "
                f"mean nearest-neighbour distance={mean_nn:.3f}, DCR risk={dcr_risk:.3f}."
            ),
            metrics={
                "exact_duplicate_rate": exact_dup_rate,
                "mean_nn_distance": mean_nn,
                "dcr_risk": dcr_risk,
                "privacy_risk": risk,
            },
            recommendations=recs,
        )

    @staticmethod
    def _exact_duplicate_rate(syn: pd.DataFrame, ref: pd.DataFrame) -> float:
        ref_keys = set(map(tuple, ref.astype(str).to_numpy().tolist()))
        if not len(syn):
            return 0.0
        hits = sum(1 for row in syn.astype(str).to_numpy().tolist() if tuple(row) in ref_keys)
        return float(hits) / float(len(syn))

    @staticmethod
    def _dcr(syn: pd.DataFrame, ref: pd.DataFrame, num: list[str]) -> tuple[float, float]:
        """Return (mean nearest-neighbour distance, risk in [0,1]).

        Columns are standardized by the reference's mean/std so distances are
        scale-free; risk is 1 when synthetic rows coincide with real ones and
        decays as they separate (risk = exp(-mean_distance)).
        """
        if len(num) == 0 or len(syn) == 0 or len(ref) == 0:
            return 0.0, 0.0
        r = ref[num].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        s = syn[num].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        mu = r.mean(axis=0)
        sd = r.std(axis=0)
        sd[sd == 0] = 1.0
        r = (r - mu) / sd
        s = (s - mu) / sd
        nn = NearestNeighbors(n_neighbors=1).fit(r)
        dist, _ = nn.kneighbors(s)
        mean_nn = float(np.mean(dist))
        risk = clamp01(float(np.exp(-mean_nn)))
        return mean_nn, risk
