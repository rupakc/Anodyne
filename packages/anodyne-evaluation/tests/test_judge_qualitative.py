from __future__ import annotations

from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.qualitative import QualitativeError, QualitativeJudge
from anodyne_evaluation.ports import EvaluationContext


class _FakeProvider:
    """Injected fake `LLMProvider` -- no network. Records the prompt it received."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(content=self._content, usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _cfg() -> ModelConfig:
    return ModelConfig(id=uuid4(), tenant_id=uuid4(), name="c", provider="openai", model="gpt-4o")


async def test_parses_rubric_and_scores() -> None:
    provider = _FakeProvider(
        '```json\n{"realism": 4, "coherence": 5, "task_fit": 3, '
        '"rationale": "mostly plausible"}\n```'
    )
    judge = QualitativeJudge(provider, _cfg())  # type: ignore[arg-type]
    ctx = EvaluationContext(subject=pd.DataFrame({"name": ["Ann"], "age": [30]}))
    score = await judge.evaluate(ctx)
    assert score.score == pytest.approx((4 + 5 + 3) / 15.0)
    assert score.metrics == {"realism": 4.0, "coherence": 5.0, "task_fit": 3.0}
    assert "plausible" in score.rationale
    # The rendered sample made it into the prompt (no network was used).
    assert provider.last_request is not None
    assert "Ann" in provider.last_request.messages[-1].content
    # Deterministic scoring: the judge calls the LLM with temperature=0.
    assert provider.last_request.params.get("temperature") == 0


async def test_low_scores_produce_recommendation() -> None:
    provider = _FakeProvider('{"realism": 1, "coherence": 2, "task_fit": 2, "rationale": "poor"}')
    score = await QualitativeJudge(provider, _cfg()).evaluate(  # type: ignore[arg-type]
        EvaluationContext(subject=pd.DataFrame({"a": [1]}))
    )
    assert score.recommendations


async def test_malformed_output_raises() -> None:
    with pytest.raises(QualitativeError):
        await QualitativeJudge(_FakeProvider("not json"), _cfg()).evaluate(  # type: ignore[arg-type]
            EvaluationContext(subject=pd.DataFrame({"a": [1]}))
        )
