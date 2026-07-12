from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.judges.fidelity import FidelityJudge
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


def _frame(seed: int, shift: float = 0.0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 400
    return pd.DataFrame(
        {
            "age": rng.normal(40 + shift, 10, n),
            "income": rng.normal(50_000 + shift * 1000, 8000, n),
            "grade": rng.choice(["a", "b", "c"], n),
        }
    )


async def test_no_reference_is_not_applicable() -> None:
    with pytest.raises(JudgeNotApplicable):
        await FidelityJudge().evaluate(EvaluationContext(subject=_frame(1)))


async def test_similar_distributions_score_high() -> None:
    ctx = EvaluationContext(subject=_frame(1), reference=_frame(2))
    score = await FidelityJudge().evaluate(ctx)
    assert score.score > 0.85
    assert 0.0 <= score.metrics["ks_mean"] < 0.15


async def test_shifted_distributions_score_lower() -> None:
    close = await FidelityJudge().evaluate(
        EvaluationContext(subject=_frame(1), reference=_frame(2))
    )
    far = await FidelityJudge().evaluate(
        EvaluationContext(subject=_frame(1, shift=40.0), reference=_frame(2))
    )
    assert far.score < close.score
    assert far.metrics["ks_mean"] > close.metrics["ks_mean"]
