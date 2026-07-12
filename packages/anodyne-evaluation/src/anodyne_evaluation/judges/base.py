"""Shared helpers for the statistical (non-LLM) experts.

`StatisticalJudge` splits the port's async `evaluate` from a synchronous
`compute`: the CPU-bound math lives in `compute`, which is what the Ray runner
dispatches as a `@ray.remote` task, while `evaluate` just calls it so these
judges still satisfy the async `Judge` port for the sequential path.
"""

from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from anodyne_tabular.profiler import PandasSampleProfiler

from anodyne_evaluation.models import ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge

# One shared profiler: reuses anodyne-tabular's semantic-type inference so the
# experts classify numeric-vs-categorical columns exactly like the generation
# side does (rather than reimplementing dtype heuristics).
_PROFILER = PandasSampleProfiler()


def clamp01(x: float) -> float:
    """Clamp to [0, 1] and coerce NaN/inf to 0.0 (worst) so a degenerate metric
    never produces an out-of-range or non-finite score."""
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def categorical_columns(df: pd.DataFrame, *, max_categories: int = 50) -> list[str]:
    """Non-numeric columns (or low-cardinality numerics) treated as categorical."""
    out: list[str] = []
    for c in df.columns:
        s = df[c].dropna()
        if s.empty:
            continue
        if pd.api.types.is_numeric_dtype(s):
            continue
        if s.astype(str).nunique() <= max_categories:
            out.append(c)
    return out


def shared_columns(a: pd.DataFrame, b: pd.DataFrame) -> list[str]:
    return [c for c in a.columns if c in b.columns]


class StatisticalJudge(Judge):
    @abstractmethod
    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        """Synchronous scoring; may raise `JudgeNotApplicable`."""

    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
        return self.compute(ctx)
