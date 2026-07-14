from __future__ import annotations

import json

# Importing the package registers every provider (generic included) as a
# side effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.task_metrics.judge import TaskMetricsJudge
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable
from anodyne_evaluation.task import TaskType


class _FakeProvider:
    """Injected fake `LLMProvider` -- no network. Returns a fixed response body."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(content=self._content, usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


async def test_generic_task_type_returns_task_quality_score(model_cfg: ModelConfig) -> None:
    llm = _FakeProvider(
        json.dumps({"realism": 4, "coherence": 5, "task_fit": 4, "rationale": "ok"})
    )
    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1, 2]}), task_type=TaskType.GENERIC)
    judge = TaskMetricsJudge(llm, model_cfg)  # type: ignore[arg-type]
    score = await judge.evaluate(ctx)
    assert score.dimension == EvalDimension.TASK_QUALITY
    assert 0.0 <= score.score <= 1.0


async def test_no_task_type_is_not_applicable(model_cfg: ModelConfig) -> None:
    llm = _FakeProvider("{}")
    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1]}), task_type=None)
    judge = TaskMetricsJudge(llm, model_cfg)  # type: ignore[arg-type]
    with pytest.raises(JudgeNotApplicable):
        await judge.evaluate(ctx)


async def test_selected_metrics_disjoint_from_catalog_is_not_applicable(
    model_cfg: ModelConfig,
) -> None:
    llm = _FakeProvider("{}")
    ctx = EvaluationContext(
        subject=pd.DataFrame({"a": [1]}),
        task_type=TaskType.GENERIC,
        selected_metrics=frozenset({"not_a_real_metric"}),
    )
    judge = TaskMetricsJudge(llm, model_cfg)  # type: ignore[arg-type]
    with pytest.raises(JudgeNotApplicable):
        await judge.evaluate(ctx)


async def test_provider_task_metric_error_is_not_applicable(model_cfg: ModelConfig) -> None:
    llm = _FakeProvider("not json")
    ctx = EvaluationContext(subject=pd.DataFrame({"a": [1]}), task_type=TaskType.GENERIC)
    judge = TaskMetricsJudge(llm, model_cfg)  # type: ignore[arg-type]
    with pytest.raises(JudgeNotApplicable):
        await judge.evaluate(ctx)
