from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.judges.utility import UtilityJudge
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


def _labeled(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 300
    x1 = rng.normal(0, 1, n)
    x2 = rng.normal(0, 1, n)
    # A learnable relationship so TSTR transfers.
    label = ((x1 + x2) > 0).astype(int)
    return pd.DataFrame({"x1": x1, "x2": x2, "label": label})


async def test_requires_reference_and_target() -> None:
    df = _labeled(1)
    with pytest.raises(JudgeNotApplicable):
        await UtilityJudge().evaluate(EvaluationContext(subject=df))  # no reference
    with pytest.raises(JudgeNotApplicable):
        await UtilityJudge().evaluate(
            EvaluationContext(subject=df, reference=_labeled(2), target_field="missing")
        )


async def test_learnable_relationship_transfers() -> None:
    score = await UtilityJudge().evaluate(
        EvaluationContext(subject=_labeled(1), reference=_labeled(2), target_field="label", seed=7)
    )
    assert score.score > 0.7
    assert score.metrics["tstr_score"] > 0.7
