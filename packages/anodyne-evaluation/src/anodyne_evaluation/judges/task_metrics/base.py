"""Shared helpers for `TaskMetricProvider` implementations (sub-system F).

Every provider reaches the tenant's model strictly through the `LLMProvider`
port and needs the same three primitives: fence-stripping the model's JSON
reply, sampling the evaluation subject deterministically, and folding a
provider's full metric dict down to the mean of just the caller-selected
subset. Kept here (rather than duplicated per provider) so every task-class
module added in later tasks follows one pattern.
"""

from __future__ import annotations

import re

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
