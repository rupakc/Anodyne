from __future__ import annotations

import re
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_text.errors import TextGenerationError
from anodyne_text.generator import TextGenerator

_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="ollama", model="llama3")


class _FakeProvider(LLMProvider):
    """Returns scripted content per call, clamped to the last entry once exhausted."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.requests: list[LLMRequest] = []

    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        idx = min(len(self.requests) - 1, len(self._responses) - 1)
        return LLMResponse(content=self._responses[idx], usage=Usage())

    async def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        raise NotImplementedError
        yield ""  # pragma: no cover - unreachable, satisfies AsyncIterator typing


def _spec(
    fields: list[FieldSpec], target_rows: int, directives: dict[str, object] | None = None
) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="customer support tickets",
        modality=Modality.TEXT,
        source="description",
        fields=fields,
        target_rows=target_rows,
        directives=directives or {},
    )


_CLASSIFICATION_FIELDS = [
    FieldSpec(name="text", semantic_type=SemanticType.TEXT),
    FieldSpec(
        name="label", semantic_type=SemanticType.CATEGORICAL, constraints={"choices": ["a", "b"]}
    ),
]


def test_single_batch_produces_exact_count() -> None:
    rows = [{"text": f"ticket about issue {i}", "label": "a"} for i in range(5)]
    provider = _FakeProvider([__import__("json").dumps(rows)])
    spec = _spec(_CLASSIFICATION_FIELDS, target_rows=5)

    table = TextGenerator(provider, _CFG).generate(spec, start_row=0, count=5, seed=1)

    assert table.num_rows == 5
    assert set(table.column_names) == {"text", "label"}
    assert len(provider.requests) == 1


def test_parses_fenced_json_block() -> None:
    import json

    rows = [{"text": "a fenced row about billing", "label": "b"}]
    content = "Sure, here you go:\n```json\n" + json.dumps(rows) + "\n```"
    provider = _FakeProvider([content])
    spec = _spec(_CLASSIFICATION_FIELDS, target_rows=1)

    table = TextGenerator(provider, _CFG).generate(spec, start_row=0, count=1, seed=1)

    assert table.num_rows == 1
    assert table.column("text").to_pylist() == ["a fenced row about billing"]


def test_quality_and_dedup_filter_across_batches() -> None:
    import json

    # First batch: one empty row, one duplicate pair, one valid unique row -> 1 net valid.
    first = [
        {"text": "", "label": "a"},
        {"text": "duplicate row here", "label": "a"},
        {"text": "duplicate row here", "label": "a"},
        {"text": "first unique valid row", "label": "b"},
    ]
    # Second batch: fills the remaining rows needed.
    second = [
        {"text": "second unique valid row", "label": "a"},
        {"text": "third unique valid row", "label": "b"},
    ]
    provider = _FakeProvider([json.dumps(first), json.dumps(second)])
    spec = _spec(_CLASSIFICATION_FIELDS, target_rows=3, directives={"batch_size": 4})

    table = TextGenerator(provider, _CFG).generate(spec, start_row=0, count=3, seed=1)

    assert table.num_rows == 3
    texts = table.column("text").to_pylist()
    assert len(set(texts)) == 3
    assert "" not in texts
    assert len(provider.requests) == 2


def test_raises_after_max_attempts_when_never_parseable() -> None:
    provider = _FakeProvider(["not json at all"])
    spec = _spec(_CLASSIFICATION_FIELDS, target_rows=5, directives={"max_attempts": 3})

    with pytest.raises(TextGenerationError):
        TextGenerator(provider, _CFG).generate(spec, start_row=0, count=5, seed=1)

    assert len(provider.requests) == 3


def test_returns_short_table_when_attempts_exhausted_without_error() -> None:
    import json

    # Every batch returns the exact same duplicate row -> only 1 unique valid row ever.
    content = json.dumps([{"text": "always the same row", "label": "a"}])
    provider = _FakeProvider([content])
    spec = _spec(_CLASSIFICATION_FIELDS, target_rows=5, directives={"max_attempts": 3})

    table = TextGenerator(provider, _CFG).generate(spec, start_row=0, count=5, seed=1)

    assert table.num_rows == 1
    assert len(provider.requests) == 3


def test_same_seed_and_range_requests_same_batch_indices() -> None:
    import json

    rows = [{"text": f"row {i}", "label": "a"} for i in range(5)]
    content = json.dumps(rows)

    def _batch_indices(requests: list[LLMRequest]) -> list[int]:
        indices = []
        for req in requests:
            user_msg = next(m.content for m in req.messages if m.role == "user")
            match = re.search(r"batch #(\d+)", user_msg)
            assert match is not None
            indices.append(int(match.group(1)))
        return indices

    provider1 = _FakeProvider([content])
    spec = _spec(_CLASSIFICATION_FIELDS, target_rows=5)
    TextGenerator(provider1, _CFG).generate(spec, start_row=100, count=5, seed=42)

    provider2 = _FakeProvider([content])
    TextGenerator(provider2, _CFG).generate(spec, start_row=100, count=5, seed=42)

    assert _batch_indices(provider1.requests) == _batch_indices(provider2.requests)
