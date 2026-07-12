from __future__ import annotations

import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import Modality
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge, JudgeNotApplicable


def test_context_defaults() -> None:
    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1, 2]}))
    assert ctx.reference is None
    assert ctx.modality is Modality.TABULAR
    assert ctx.seed == 0


async def test_judge_subclass_contract() -> None:
    class OkJudge(Judge):
        dimension = EvalDimension.DIVERSITY

        async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
            return ExpertScore(dimension=self.dimension, score=0.5, rationale="x")

    class SkipJudge(Judge):
        dimension = EvalDimension.FIDELITY

        async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
            raise JudgeNotApplicable("no reference")

    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1]}))
    assert (await OkJudge().evaluate(ctx)).score == 0.5
    with pytest.raises(JudgeNotApplicable):
        await SkipJudge().evaluate(ctx)
