from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.judges.bias import BiasJudge
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


async def test_requires_sensitive_field() -> None:
    with pytest.raises(JudgeNotApplicable):
        await BiasJudge().evaluate(EvaluationContext(subject=pd.DataFrame({"a": [1, 2]})))


async def test_balanced_fair_data_scores_high() -> None:
    df = pd.DataFrame(
        {"group": ["m", "f"] * 100, "hired": [1, 1, 0, 0] * 50}  # equal outcome per group
    )
    score = await BiasJudge().evaluate(
        EvaluationContext(subject=df, sensitive_field="group", target_field="hired")
    )
    assert score.score > 0.8
    assert score.metrics["demographic_parity_diff"] < 0.1


async def test_outcome_disparity_lowers_score() -> None:
    # Group "m" always hired, group "f" never -> maximal demographic-parity diff.
    df = pd.DataFrame({"group": ["m"] * 100 + ["f"] * 100, "hired": [1] * 100 + [0] * 100})
    score = await BiasJudge().evaluate(
        EvaluationContext(subject=df, sensitive_field="group", target_field="hired")
    )
    assert score.metrics["demographic_parity_diff"] > 0.9
    assert score.score < 0.6
    assert score.recommendations
