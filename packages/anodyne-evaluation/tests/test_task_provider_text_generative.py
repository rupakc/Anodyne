from __future__ import annotations

import json
import re

# Importing the package registers every provider (qa/summarization/chat included) as a
# side effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import provider_for

# ---------------------------------------------------------------------------
# QA
# ---------------------------------------------------------------------------

_QA_QUESTIONS = [
    "What is the capital of France?",
    "Why is the sky blue?",
    "How does an engine work?",
]
_QA_ANSWERS = ["Paris", "Rayleigh scattering", ""]
_QA_CONTEXTS = [
    "France is a country in Europe. Its capital is Paris.",
    "Light scatters in the atmosphere.",
    "Engines combust fuel.",
]
# correct/grounded truth per question, independent of sample shuffle order.
_QA_CORRECT = dict(zip(_QA_QUESTIONS, [True, True, False], strict=True))
_QA_GROUNDED = dict(zip(_QA_QUESTIONS, [True, False, True], strict=True))
# 2/3 correct, 2/3 grounded.
_QA_EXPECTED_CORRECTNESS = 2 / 3
_QA_EXPECTED_GROUNDEDNESS = 2 / 3
# non-empty answers: 2 of 3.
_QA_EXPECTED_ANSWERABLE_RATE = 2 / 3
# leading interrogatives: what/why/how -> 3 distinct / 7.
_QA_EXPECTED_DIVERSITY = 3 / 7


class _FakeQAProvider:
    """Parses the `Question: ...` lines out of the prompt (order-invariant re the
    sampling shuffle) and returns the canned correct/grounded verdict for each."""

    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        prompt = request.messages[-1].content
        questions = re.findall(r"Question: (.+)", prompt)
        correct = [_QA_CORRECT[q] for q in questions]
        grounded = [_QA_GROUNDED[q] for q in questions]
        return LLMResponse(
            content=json.dumps({"correct": correct, "grounded": grounded}),
            usage=Usage(total_tokens=1),
        )

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _qa_frame() -> pd.DataFrame:
    return pd.DataFrame({"question": _QA_QUESTIONS, "answer": _QA_ANSWERS, "context": _QA_CONTEXTS})


_QA_ALL_SELECTED = frozenset(
    {"answer_correctness", "groundedness", "answerable_rate", "question_type_diversity"}
)


async def test_qa_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    llm = _FakeQAProvider()
    ctx = EvaluationContext(subject=_qa_frame(), task_type=TaskType.QA, sample_rows=3)
    prov = provider_for(TaskType.QA)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_QA_ALL_SELECTED)  # type: ignore[arg-type]
    assert score.dimension.value == "task_quality"
    assert score.metrics["answer_correctness"] == pytest.approx(_QA_EXPECTED_CORRECTNESS)
    assert score.metrics["groundedness"] == pytest.approx(_QA_EXPECTED_GROUNDEDNESS)
    assert score.metrics["answerable_rate"] == pytest.approx(_QA_EXPECTED_ANSWERABLE_RATE)
    assert score.metrics["question_type_diversity"] == pytest.approx(_QA_EXPECTED_DIVERSITY)
    expected_score = sum(score.metrics[k] for k in _QA_ALL_SELECTED) / len(_QA_ALL_SELECTED)
    assert score.score == pytest.approx(expected_score)
    assert llm.last_request is not None
    assert llm.last_request.params.get("temperature") == 0


async def test_qa_provider_skips_llm_when_not_selected(model_cfg: ModelConfig) -> None:
    class _ExplodingProvider:
        async def complete(self, config, request):  # type: ignore[no-untyped-def]
            raise AssertionError("LLM should not be called when no LLM metric is selected")

        def stream(self, config, request): ...  # type: ignore[no-untyped-def]

    ctx = EvaluationContext(subject=_qa_frame(), task_type=TaskType.QA, sample_rows=3)
    prov = provider_for(TaskType.QA)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"answerable_rate", "question_type_diversity"}),
    )
    assert set(score.metrics) == {"answerable_rate", "question_type_diversity"}


async def test_qa_provider_missing_answer_column_raises(model_cfg: ModelConfig) -> None:
    ctx = EvaluationContext(
        subject=pd.DataFrame({"question": _QA_QUESTIONS}),
        task_type=TaskType.QA,
        sample_rows=3,
    )
    prov = provider_for(TaskType.QA)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _FakeQAProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"answerable_rate"}),
        )


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------

# document1/summary1: ratio 11/22 = 0.5; bigram overlap 2/2 = 1.0.
_DOCUMENT_1 = "the cat sat on the mat"
_SUMMARY_1 = "the cat sat"
# document2/summary2: ratio 31/12 (>1, capped to 1.0); bigram overlap 2/5 = 0.4.
_DOCUMENT_2 = "dog ran fast"
_SUMMARY_2 = "dog ran fast now really quickly"
# mean compression_ratio = (0.5 + 1.0) / 2 = 0.75
_EXPECTED_COMPRESSION_RATIO = 0.75
# mean bigram overlap = (1.0 + 0.4) / 2 = 0.7 -> abstractiveness = 1 - 0.7 = 0.3
_EXPECTED_ABSTRACTIVENESS = 0.3


class _FakeSummarizationProvider:
    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(
            content=json.dumps({"faithfulness": 4, "coverage": 5, "conciseness": 4}),
            usage=Usage(total_tokens=1),
        )

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _summarization_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "document": [_DOCUMENT_1, _DOCUMENT_2],
            "summary": [_SUMMARY_1, _SUMMARY_2],
        }
    )


_SUMM_ALL_SELECTED = frozenset(
    {"faithfulness", "coverage", "conciseness", "compression_ratio", "abstractiveness"}
)


async def test_summarization_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    llm = _FakeSummarizationProvider()
    ctx = EvaluationContext(
        subject=_summarization_frame(), task_type=TaskType.SUMMARIZATION, sample_rows=2
    )
    prov = provider_for(TaskType.SUMMARIZATION)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_SUMM_ALL_SELECTED)  # type: ignore[arg-type]
    assert score.dimension.value == "task_quality"
    assert score.metrics["faithfulness"] == pytest.approx(0.8)
    assert score.metrics["coverage"] == pytest.approx(1.0)
    assert score.metrics["conciseness"] == pytest.approx(0.8)
    assert score.metrics["compression_ratio"] == pytest.approx(_EXPECTED_COMPRESSION_RATIO)
    assert score.metrics["abstractiveness"] == pytest.approx(_EXPECTED_ABSTRACTIVENESS)
    expected_score = sum(score.metrics[k] for k in _SUMM_ALL_SELECTED) / len(_SUMM_ALL_SELECTED)
    assert score.score == pytest.approx(expected_score)
    assert llm.last_request is not None
    assert llm.last_request.params.get("temperature") == 0


async def test_summarization_provider_skips_llm_when_not_selected(model_cfg: ModelConfig) -> None:
    class _ExplodingProvider:
        async def complete(self, config, request):  # type: ignore[no-untyped-def]
            raise AssertionError("LLM should not be called when no LLM metric is selected")

        def stream(self, config, request): ...  # type: ignore[no-untyped-def]

    ctx = EvaluationContext(
        subject=_summarization_frame(), task_type=TaskType.SUMMARIZATION, sample_rows=2
    )
    prov = provider_for(TaskType.SUMMARIZATION)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"compression_ratio", "abstractiveness"}),
    )
    assert set(score.metrics) == {"compression_ratio", "abstractiveness"}


async def test_summarization_provider_missing_summary_column_raises(
    model_cfg: ModelConfig,
) -> None:
    ctx = EvaluationContext(
        subject=pd.DataFrame({"document": [_DOCUMENT_1, _DOCUMENT_2]}),
        task_type=TaskType.SUMMARIZATION,
        sample_rows=2,
    )
    prov = provider_for(TaskType.SUMMARIZATION)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _FakeSummarizationProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"compression_ratio"}),
        )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------

_CHAT_INSTRUCTIONS = [
    "Write a haiku about the ocean.",
    "Summarize this article.",
    "What's 2+2?",
]
_CHAT_RESPONSES = [
    "Waves crash on the shore...",
    "The article discusses X and Y.",
    "4",
]


class _FakeChatProvider:
    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(
            content=json.dumps({"instruction_following": 5, "coherence": 4}),
            usage=Usage(total_tokens=1),
        )

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _chat_frame() -> pd.DataFrame:
    return pd.DataFrame({"instruction": _CHAT_INSTRUCTIONS, "response": _CHAT_RESPONSES})


_CHAT_ALL_SELECTED = frozenset({"instruction_following", "coherence", "turn_validity"})


async def test_chat_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    llm = _FakeChatProvider()
    ctx = EvaluationContext(subject=_chat_frame(), task_type=TaskType.CHAT, sample_rows=3)
    prov = provider_for(TaskType.CHAT)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_CHAT_ALL_SELECTED)  # type: ignore[arg-type]
    assert score.dimension.value == "task_quality"
    assert score.metrics["instruction_following"] == pytest.approx(1.0)
    assert score.metrics["coherence"] == pytest.approx(0.8)
    assert score.metrics["turn_validity"] == pytest.approx(1.0)
    expected_score = sum(score.metrics[k] for k in _CHAT_ALL_SELECTED) / len(_CHAT_ALL_SELECTED)
    assert score.score == pytest.approx(expected_score)
    assert llm.last_request is not None
    assert llm.last_request.params.get("temperature") == 0


async def test_chat_provider_turn_validity_counts_empty_turns(model_cfg: ModelConfig) -> None:
    df = pd.DataFrame(
        {
            "instruction": ["Do X.", "", "Do Z."],
            "response": ["Done.", "Ok.", ""],
        }
    )
    ctx = EvaluationContext(subject=df, task_type=TaskType.CHAT, sample_rows=3)
    prov = provider_for(TaskType.CHAT)
    assert prov is not None

    class _ExplodingProvider:
        async def complete(self, config, request):  # type: ignore[no-untyped-def]
            raise AssertionError("LLM should not be called when no LLM metric is selected")

        def stream(self, config, request): ...  # type: ignore[no-untyped-def]

    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"turn_validity"}),
    )
    # Only the first row has both a non-empty instruction and response.
    assert score.metrics["turn_validity"] == pytest.approx(1 / 3)


async def test_chat_provider_missing_response_column_raises(model_cfg: ModelConfig) -> None:
    ctx = EvaluationContext(
        subject=pd.DataFrame({"instruction": _CHAT_INSTRUCTIONS}),
        task_type=TaskType.CHAT,
        sample_rows=3,
    )
    prov = provider_for(TaskType.CHAT)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _FakeChatProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"turn_validity"}),
        )
