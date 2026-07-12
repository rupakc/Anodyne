from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.judges.privacy import PrivacyJudge
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({"x": rng.normal(0, 1, 200), "y": rng.normal(5, 2, 200)})


async def test_no_reference_is_not_applicable() -> None:
    with pytest.raises(JudgeNotApplicable):
        await PrivacyJudge().evaluate(EvaluationContext(subject=_frame(1)))


async def test_verbatim_copy_is_high_risk() -> None:
    ref = _frame(1)
    score = await PrivacyJudge().evaluate(EvaluationContext(subject=ref.copy(), reference=ref))
    assert score.metrics["exact_duplicate_rate"] == 1.0
    assert score.score < 0.2  # high leakage -> low score


async def test_independent_data_is_low_risk() -> None:
    score = await PrivacyJudge().evaluate(
        EvaluationContext(subject=_frame(1), reference=_frame(99))
    )
    assert score.metrics["exact_duplicate_rate"] == 0.0
    assert score.score > 0.5
