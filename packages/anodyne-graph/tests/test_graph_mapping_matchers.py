from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_graph.mapping.matchers import (
    EmbeddingMatcher,
    LexicalMatcher,
    LLMMatcher,
)
from anodyne_graph.mapping.models import MappingRelation
from anodyne_graph.mapping.ports import EntityMatcher

_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="fake", model="f")


class _Provider(LLMProvider):
    def __init__(self, content: str) -> None:
        self._c = content
        self.calls = 0

    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        self.calls += 1
        return LLMResponse(content=self._c, usage=Usage())

    async def _s(self) -> AsyncIterator[str]:
        if False:
            yield ""

    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        return self._s()


# --- LexicalMatcher ---------------------------------------------------------


def test_lexical_identical_is_one() -> None:
    lex = LexicalMatcher()
    assert lex.score("Person", "Person") == 1.0


def test_lexical_case_and_style_insensitive_tokenization() -> None:
    lex = LexicalMatcher()
    # camelCase / snake / case differences normalize to the same tokens.
    assert lex.score("WorksAt", "works_at") == 1.0
    assert lex.score("Person", "person") == 1.0


def test_lexical_disjoint_short_strings_is_zero() -> None:
    lex = LexicalMatcher()
    assert lex.score("Cat", "Dog") == 0.0


def test_lexical_partial_overlap_is_deterministic_value() -> None:
    lex = LexicalMatcher()
    # tokens {has, employee} vs {has, employer}: jaccard=1/3;
    # edit("has employee","has employer")=1 over len 12 -> 11/12.
    expected = 0.5 * (1.0 / 3.0) + 0.5 * (11.0 / 12.0)
    assert lex.score("HasEmployee", "HasEmployer") == pytest.approx(expected)


def test_lexical_satisfies_entity_matcher_protocol() -> None:
    assert isinstance(LexicalMatcher(), EntityMatcher)


# --- EmbeddingMatcher -------------------------------------------------------


def _fake_embed(vectors: dict[str, list[float]]) -> object:
    calls: list[int] = [0]

    def embed(texts: Sequence[str]) -> list[list[float]]:
        calls[0] += 1
        return [vectors[t] for t in texts]

    embed.calls = calls  # type: ignore[attr-defined]
    return embed


def test_embedding_cosine_mapped_to_unit_interval() -> None:
    embed = _fake_embed({"Person": [1.0, 0.0], "Human": [1.0, 0.0], "Company": [0.0, 1.0]})
    em = EmbeddingMatcher(embed)  # type: ignore[arg-type]
    assert em.available is True
    assert em.score("Person", "Human") == pytest.approx(1.0)  # cos 1 -> 1.0
    assert em.score("Person", "Company") == pytest.approx(0.5)  # cos 0 -> 0.5


def test_embedding_caches_embeddings() -> None:
    embed = _fake_embed({"Person": [1.0, 0.0], "Human": [1.0, 0.0]})
    em = EmbeddingMatcher(embed)  # type: ignore[arg-type]
    em.score("Person", "Human")
    em.score("Person", "Human")
    # Both strings embedded once total, then served from cache.
    assert embed.calls[0] <= 2  # type: ignore[attr-defined]


def test_embedding_degrades_gracefully_without_fn() -> None:
    em = EmbeddingMatcher(None)
    assert em.available is False
    assert em.score("Person", "Human") == 0.0


# --- LLMMatcher -------------------------------------------------------------


async def test_llm_matcher_parses_predicate_confidence_justification() -> None:
    payload = json.dumps(
        {"predicate": "close_match", "confidence": 0.72, "justification": "synonyms"}
    )
    provider = _Provider(f"```json\n{payload}\n```")
    llm = LLMMatcher(provider, _CFG)
    j = await llm.adjudicate("Person", "Human", "a person", "a human being")
    assert j.predicate == MappingRelation.CLOSE_MATCH
    assert j.confidence == pytest.approx(0.72)
    assert j.justification == "synonyms"
    assert provider.calls == 1


async def test_llm_matcher_malformed_output_is_low_confidence_fallback() -> None:
    provider = _Provider("not json at all")
    llm = LLMMatcher(provider, _CFG)
    j = await llm.adjudicate("Person", "Widget", "", "")
    assert j.confidence == 0.0
    assert j.predicate == MappingRelation.RELATED_MATCH


async def test_llm_matcher_clamps_and_rejects_bad_predicate() -> None:
    payload = json.dumps({"predicate": "bogus", "confidence": 2.0, "justification": "x"})
    provider = _Provider(payload)
    llm = LLMMatcher(provider, _CFG)
    j = await llm.adjudicate("A", "B", "", "")
    assert 0.0 <= j.confidence <= 1.0
    assert j.predicate == MappingRelation.RELATED_MATCH
