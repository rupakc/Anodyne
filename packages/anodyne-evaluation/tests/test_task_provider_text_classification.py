from __future__ import annotations

import json

# Importing the package registers every provider (text_classification included) as a
# side effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import provider_for

# text -> true label; "a" is deliberately mispredicted below to force accuracy = 0.75.
_TEXTS = ["a", "b", "c", "d"]
_LABELS = ["pos", "neg", "pos", "neg"]
_TRUE = dict(zip(_TEXTS, _LABELS, strict=True))
_WRONG_TEXT = "a"  # predicted "neg" instead of the true "pos"


class _FakeProvider:
    """Injected fake `LLMProvider` -- parses the numbered texts out of the prompt so
    the response lines up regardless of the sampling shuffle order, then returns the
    correct label for each except `_WRONG_TEXT`, which is flipped."""

    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        prompt = request.messages[-1].content
        ordered_texts = [
            line.split(". ", 1)[1]
            for line in prompt.splitlines()
            if line and line[0].isdigit() and ". " in line
        ]
        predicted = []
        for t in ordered_texts:
            true_label = _TRUE[t]
            if t == _WRONG_TEXT:
                predicted.append("neg" if true_label == "pos" else "pos")
            else:
                predicted.append(true_label)
        return LLMResponse(content=json.dumps({"labels": predicted}), usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _frame() -> pd.DataFrame:
    return pd.DataFrame({"text": _TEXTS, "label": _LABELS})


_ALL_SELECTED = frozenset({"accuracy", "macro_f1", "class_balance", "duplicate_rate"})
_EXPECTED_MACRO_F1 = 11.0 / 15.0  # hand-computed for the a/b/c/d confusion above


async def test_text_classification_provider_scores_all_metrics(
    model_cfg: ModelConfig,
) -> None:
    llm = _FakeProvider()
    ctx = EvaluationContext(subject=_frame(), task_type=TaskType.TEXT_CLASSIFICATION, sample_rows=4)
    prov = provider_for(TaskType.TEXT_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(
        ctx,
        llm,  # type: ignore[arg-type]
        model_cfg,
        selected=_ALL_SELECTED,
    )
    assert score.dimension.value == "task_quality"
    assert score.metrics["accuracy"] == pytest.approx(0.75)
    assert score.metrics["class_balance"] == pytest.approx(1.0)
    assert score.metrics["duplicate_rate"] == pytest.approx(0.0)
    assert score.metrics["macro_f1"] == pytest.approx(_EXPECTED_MACRO_F1)
    expected_score = sum(score.metrics[k] for k in _ALL_SELECTED) / len(_ALL_SELECTED)
    assert score.score == pytest.approx(expected_score)
    assert llm.last_request is not None
    assert llm.last_request.params.get("temperature") == 0


async def test_text_classification_provider_catalog() -> None:
    prov = provider_for(TaskType.TEXT_CLASSIFICATION)
    assert prov is not None
    catalog = prov.metric_catalog()
    keys = {m.key for m in catalog}
    assert keys == {"accuracy", "macro_f1", "class_balance", "duplicate_rate"}
    llm_keys = {m.key for m in catalog if m.requires_llm}
    assert llm_keys == {"accuracy", "macro_f1"}


async def test_text_classification_provider_skips_llm_when_not_selected(
    model_cfg: ModelConfig,
) -> None:
    """If neither accuracy nor macro_f1 is selected, no LLM call should happen."""

    class _ExplodingProvider:
        async def complete(self, config, request):  # type: ignore[no-untyped-def]
            raise AssertionError("LLM should not be called when accuracy/macro_f1 unselected")

        def stream(self, config, request): ...  # type: ignore[no-untyped-def]

    ctx = EvaluationContext(subject=_frame(), task_type=TaskType.TEXT_CLASSIFICATION, sample_rows=4)
    prov = provider_for(TaskType.TEXT_CLASSIFICATION)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"class_balance", "duplicate_rate"}),
    )
    assert set(score.metrics) == {"class_balance", "duplicate_rate"}
    assert score.score == pytest.approx(0.5)  # (1.0 + 0.0) / 2


async def test_text_classification_provider_missing_label_column_raises(
    model_cfg: ModelConfig,
) -> None:
    llm = _FakeProvider()
    ctx = EvaluationContext(
        subject=pd.DataFrame({"text": _TEXTS}),
        task_type=TaskType.TEXT_CLASSIFICATION,
        sample_rows=4,
    )
    prov = provider_for(TaskType.TEXT_CLASSIFICATION)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            llm,  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"class_balance"}),
        )
