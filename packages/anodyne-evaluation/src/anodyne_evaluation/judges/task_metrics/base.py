"""Shared helpers for `TaskMetricProvider` implementations (sub-system F).

Every provider reaches the tenant's model strictly through the `LLMProvider`
port and needs the same three primitives: fence-stripping the model's JSON
reply, sampling the evaluation subject deterministically, and folding a
provider's full metric dict down to the mean of just the caller-selected
subset. Kept here (rather than duplicated per provider) so every task-class
module added in later tasks follows one pattern.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable

import pandas as pd  # type: ignore[import-untyped]

from anodyne_evaluation.ports import EvaluationContext

_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


class TaskMetricError(Exception):
    """Raised when a provider's LLM output can't be parsed."""


def strip_json(raw: str) -> str:
    text = raw.strip()
    m = _FENCE.search(text)
    return m.group(1).strip() if m else text


def sample_frame(ctx: EvaluationContext) -> pd.DataFrame:
    n = min(ctx.sample_rows, len(ctx.subject))
    if n <= 0:
        return ctx.subject.head(0)
    return ctx.subject.sample(n=n, random_state=ctx.seed).reset_index(drop=True)


def mean_contribution(metrics: dict[str, float], selected: frozenset[str]) -> float:
    vals = [metrics[k] for k in metrics if k in selected]
    return sum(vals) / len(vals) if vals else 0.0


def text_value(x: object) -> str:
    """Coerce a row value to a string, treating `None`/NaN as `""`."""
    if x is None:
        return ""
    try:
        if pd.isna(x):
            return ""
    except (TypeError, ValueError):
        pass
    return str(x)


def is_nonempty(x: object) -> bool:
    """`True` iff `x` is not `None`/NaN and its stripped string form is non-empty."""
    if x is None:
        return False
    try:
        if pd.isna(x):
            return False
    except (TypeError, ValueError):
        pass
    return str(x).strip() != ""


def normalized_label_entropy(labels: pd.Series | Iterable) -> float:  # type: ignore[type-arg]
    """Normalized Shannon entropy (H / log(k)) of a label distribution.

    `1.0` for a perfectly uniform distribution, `0.0` when fewer than 2 distinct
    labels are present. Shared by every `class_balance`/`label_balance` metric
    across task-metric providers -- do not fork this per-provider.
    """
    counts = pd.Series(labels).value_counts()
    k = counts.shape[0]
    if k < 2:
        return 0.0
    total = counts.sum()
    probs = [c / total for c in counts if c > 0]
    h = -sum(p * math.log(p) for p in probs)
    return float(h / math.log(k))
