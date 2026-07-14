from __future__ import annotations

import json

# Importing the package registers every provider (generic included) as a
# side effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import provider_for


class _FakeProvider:
    """Injected fake `LLMProvider` -- no network. Records the prompt it received."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(content=self._content, usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


async def test_generic_provider_scores_rubric(model_cfg: ModelConfig) -> None:
    llm = _FakeProvider(
        json.dumps({"realism": 4, "coherence": 5, "task_fit": 4, "rationale": "ok"})
    )
    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1, 2]}), task_type=TaskType.GENERIC)
    prov = provider_for(TaskType.GENERIC)
    assert prov is not None
    score = await prov.score(
        ctx,
        llm,  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"realism", "coherence", "task_fit"}),
    )
    assert 0.0 <= score.score <= 1.0
    assert score.score == pytest.approx((4 + 5 + 4) / 15.0)
    assert set(score.metrics) >= {"realism", "coherence", "task_fit"}
    assert score.dimension.value == "task_quality"
    assert llm.last_request is not None
    assert llm.last_request.params.get("temperature") == 0


async def test_generic_provider_catalog_has_three_llm_metrics() -> None:
    prov = provider_for(TaskType.GENERIC)
    assert prov is not None
    catalog = prov.metric_catalog()
    keys = {m.key for m in catalog}
    assert keys == {"realism", "coherence", "task_fit"}
    assert all(m.requires_llm for m in catalog)


async def test_generic_provider_unparseable_output_raises(model_cfg: ModelConfig) -> None:
    from anodyne_evaluation.judges.task_metrics.base import TaskMetricError

    llm = _FakeProvider("not json")
    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1]}), task_type=TaskType.GENERIC)
    prov = provider_for(TaskType.GENERIC)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            llm,  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"realism"}),
        )
