"""ML-utility expert: Train-on-Synthetic, Test-on-Real (TSTR).

Requires a reference dataset + a `target_field` (else `JudgeNotApplicable`; no
LLM). Trains a small, seeded sklearn model on the synthetic data and scores it
on the real data (TSTR), then compares to a real-trained/real-tested baseline
(TRTR). The efficacy ratio TSTR/TRTR approximates how well models learned on
the synthetic data transfer to reality. Higher score == more useful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from sklearn.ensemble import (  # type: ignore[import-untyped]
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.metrics import accuracy_score, r2_score  # type: ignore[import-untyped]

from anodyne_evaluation.judges.base import StatisticalJudge, clamp01, numeric_columns
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


class UtilityJudge(StatisticalJudge):
    dimension = EvalDimension.UTILITY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        if ctx.reference is None:
            raise JudgeNotApplicable("utility (TSTR) requires a reference dataset")
        target = ctx.target_field
        if not target or target not in ctx.subject.columns or target not in ctx.reference.columns:
            raise JudgeNotApplicable(
                "utility (TSTR) requires a target_field present in both datasets"
            )

        syn, ref = ctx.subject, ctx.reference
        features = [c for c in numeric_columns(syn) if c in ref.columns and c != target]
        if not features:
            raise JudgeNotApplicable("utility (TSTR) requires numeric feature columns")

        classify = self._is_classification(ref[target])
        xs, ys = self._xy(syn, features, target)
        xr, yr = self._xy(ref, features, target)
        if len(xs) < 2 or len(xr) < 2:
            raise JudgeNotApplicable("utility (TSTR) requires at least two rows per dataset")

        tstr = self._fit_score(xs, ys, xr, yr, classify, ctx.seed)
        trtr = self._fit_score(xr, yr, xr, yr, classify, ctx.seed)

        ratio = clamp01(tstr / trtr) if trtr > 0 else 0.0
        recs: list[str] = []
        if ratio < 0.7:
            recs.append(
                "Models trained on synthetic transfer poorly to real data; "
                "improve feature-target relationships."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=ratio,
            rationale=(
                f"TSTR efficacy on target '{target}' "
                f"({'classification' if classify else 'regression'}): "
                f"TSTR={tstr:.3f}, TRTR={trtr:.3f}, ratio={ratio:.3f}."
            ),
            metrics={"tstr_score": tstr, "trtr_score": trtr, "efficacy_ratio": ratio},
            recommendations=recs,
        )

    @staticmethod
    def _is_classification(y: pd.Series) -> bool:
        if not pd.api.types.is_numeric_dtype(y):
            return True
        return bool(y.dropna().nunique() <= 10)

    @staticmethod
    def _xy(df: pd.DataFrame, features: list[str], target: str) -> tuple[np.ndarray, np.ndarray]:
        sub = df[[*features, target]].dropna()
        x = sub[features].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        y = sub[target].to_numpy()
        return x, y

    @staticmethod
    def _fit_score(
        xtr: np.ndarray,
        ytr: np.ndarray,
        xte: np.ndarray,
        yte: np.ndarray,
        classify: bool,
        seed: int,
    ) -> float:
        if classify:
            if len(np.unique(ytr)) < 2:
                # Degenerate single-class training set: fall back to the base rate.
                pred = np.full(shape=len(yte), fill_value=ytr[0])
                return float(accuracy_score(yte.astype(str), pred.astype(str)))
            model = RandomForestClassifier(n_estimators=25, random_state=seed)
            model.fit(xtr, ytr.astype(str))
            return float(accuracy_score(yte.astype(str), model.predict(xte)))
        model = RandomForestRegressor(n_estimators=25, random_state=seed)
        model.fit(xtr, ytr.astype(float))
        return float(max(0.0, r2_score(yte.astype(float), model.predict(xte))))
