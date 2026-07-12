"""RayJudgeRunner integration test: needs a local Ray runtime (like the other
compute ray tests). Verifies the statistical experts fan out over Ray and the
async qualitative expert runs inline, producing the same set of scores the
sequential runner would."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
import pytest
import ray
from anodyne_compute.ray_evaluation import RayJudgeRunner
from anodyne_core.models import LLMResponse, ModelConfig, Usage
from anodyne_evaluation.evaluator import default_judges
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import EvaluationContext

pytestmark = pytest.mark.integration


class _FakeProvider:
    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        return LLMResponse(
            content='{"realism": 4, "coherence": 4, "task_fit": 4, "rationale": "ok"}',
            usage=Usage(total_tokens=1),
        )

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _frame(seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 1, 120)
    return pd.DataFrame(
        {"x": x, "label": (x > 0).astype(int), "group": rng.choice(["a", "b"], 120)}
    )


async def test_ray_runner_matches_dimension_set() -> None:
    if not ray.is_initialized():
        ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    cfg = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="c", provider="openai", model="gpt-4o")
    judges = default_judges(_FakeProvider(), cfg)  # type: ignore[arg-type]
    ctx = EvaluationContext(
        subject=_frame(1),
        reference=_frame(2),
        target_field="label",
        sensitive_field="group",
        seed=3,
    )
    scores = await RayJudgeRunner()(judges, ctx)
    dims = {s.dimension for s in scores}
    assert dims == {
        EvalDimension.FIDELITY,
        EvalDimension.DIVERSITY,
        EvalDimension.PRIVACY,
        EvalDimension.UTILITY,
        EvalDimension.BIAS,
        EvalDimension.QUALITATIVE,
    }
