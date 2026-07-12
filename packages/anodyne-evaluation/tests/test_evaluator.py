from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from anodyne_evaluation.evaluator import MoEEvaluator, default_judges
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import EvaluationContext


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 200
    x = rng.normal(0, 1, n)
    return pd.DataFrame(
        {
            "x": x,
            "y": rng.normal(3, 1, n),
            "label": (x > 0).astype(int),
            "group": rng.choice(["a", "b"], n),
        }
    )


async def test_end_to_end_sequential_produces_360_report() -> None:
    # No LLM provider -> qualitative expert is omitted; the five statistical
    # experts run. With reference + target + sensitive field, all five apply.
    evaluator = MoEEvaluator(default_judges())
    ctx = EvaluationContext(
        subject=_frame(1),
        reference=_frame(2),
        target_field="label",
        sensitive_field="group",
        seed=3,
    )
    report = await evaluator.evaluate(
        ctx,
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        reference_version_id=uuid4(),
    )
    dims = {s.dimension for s in report.expert_scores}
    assert dims == {
        EvalDimension.FIDELITY,
        EvalDimension.DIVERSITY,
        EvalDimension.PRIVACY,
        EvalDimension.UTILITY,
        EvalDimension.BIAS,
    }
    assert 0.0 <= report.overall_score <= 1.0
    assert sum(report.weights.values()) > 0.99


async def test_reference_free_run_skips_reference_experts() -> None:
    # Without a reference, fidelity/privacy/utility drop out; diversity remains.
    report = await MoEEvaluator(default_judges()).evaluate(
        EvaluationContext(subject=_frame(1)),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
    )
    dims = {s.dimension for s in report.expert_scores}
    assert EvalDimension.DIVERSITY in dims
    assert EvalDimension.FIDELITY not in dims
    assert EvalDimension.PRIVACY not in dims
