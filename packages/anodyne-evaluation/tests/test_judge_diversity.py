from __future__ import annotations

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from anodyne_evaluation.judges.diversity import DiversityJudge
from anodyne_evaluation.ports import EvaluationContext


async def test_diverse_data_scores_higher_than_collapsed() -> None:
    rng = np.random.default_rng(0)
    diverse = pd.DataFrame(
        {"cat": rng.choice(list("abcdef"), 300), "val": rng.integers(0, 100, 300)}
    )
    collapsed = pd.DataFrame({"cat": ["a"] * 295 + ["b"] * 5, "val": [7] * 300})

    d = await DiversityJudge().evaluate(EvaluationContext(subject=diverse))
    c = await DiversityJudge().evaluate(EvaluationContext(subject=collapsed))

    assert d.score > c.score
    assert c.metrics["max_mode_freq"] > 0.9
    assert d.metrics["mean_entropy"] > 0.8
