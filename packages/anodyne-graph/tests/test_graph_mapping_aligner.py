from __future__ import annotations

import json
from collections.abc import AsyncIterator, Sequence
from uuid import uuid4

from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_graph.mapping.aligner import AlignmentThresholds, OntologyAligner
from anodyne_graph.mapping.matchers import EmbeddingMatcher, LexicalMatcher, LLMMatcher
from anodyne_graph.mapping.models import MappingRelation, MappingSet
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec

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


def _source() -> GraphOntology:
    return GraphOntology(
        node_types=[
            NodeType(name="Person", properties=[PropertySpec(name="name")]),
            NodeType(name="Organization", properties=[PropertySpec(name="name")]),
            NodeType(name="Gadget", properties=[PropertySpec(name="sku")]),
        ],
        edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Organization")],
    )


def _target() -> GraphOntology:
    return GraphOntology(
        node_types=[
            NodeType(name="Person", properties=[PropertySpec(name="name")]),
            NodeType(name="Organisation", properties=[PropertySpec(name="name")]),
            NodeType(name="Vehicle", properties=[PropertySpec(name="vin")]),
        ],
        edge_types=[
            EdgeType(name="WORKS_AT", source_type="Person", target_type="Organisation"),
            EdgeType(name="EMPLOYED_BY", source_type="Person", target_type="Organisation"),
        ],
    )


_THRESHOLDS = AlignmentThresholds(auto_accept=0.9, review_floor=0.4, prefilter_floor=0.3)


async def _align() -> MappingSet:
    aligner = OntologyAligner(LexicalMatcher(), thresholds=_THRESHOLDS)
    return await aligner.align(_source(), _target(), source_id="src", target_id="tgt")


async def test_threshold_routing_lexical_only() -> None:
    ms = await _align()
    by_subj = {m.subject_id: m for m in ms.mappings}
    # exact label -> auto-accepted exact_match, not flagged.
    assert by_subj["Person"].predicate == MappingRelation.EXACT_MATCH
    assert by_subj["Person"].needs_review is False
    assert by_subj["Person"].object_id == "Person"
    # near-miss label -> review band -> flagged.
    assert by_subj["Organization"].object_id == "Organisation"
    assert by_subj["Organization"].needs_review is True
    # edge types are aligned too.
    assert by_subj["WORKS_AT"].object_id == "WORKS_AT"
    assert by_subj["WORKS_AT"].needs_review is False
    # dissimilar source entity is dropped entirely.
    assert "Gadget" not in by_subj


async def test_output_sorted_and_metadata_recorded() -> None:
    ms = await _align()
    subjects = [(m.subject_id, m.object_id) for m in ms.mappings]
    assert subjects == sorted(subjects)
    assert ms.source_ontology_id == "src"
    assert ms.metadata["auto_accept"] == 0.9
    assert ms.metadata["review_floor"] == 0.4
    assert "seed" in ms.metadata


async def test_deterministic_across_two_runs() -> None:
    a = await _align()
    b = await _align()
    assert a.model_dump() == b.model_dump()


async def test_llm_adjudicates_only_borderline_pairs() -> None:
    payload = json.dumps(
        {"predicate": "exact_match", "confidence": 0.96, "justification": "same concept"}
    )
    provider = _Provider(payload)
    aligner = OntologyAligner(
        LexicalMatcher(),
        llm=LLMMatcher(provider, _CFG),
        thresholds=_THRESHOLDS,
    )
    ms = await aligner.align(_source(), _target(), source_id="src", target_id="tgt")
    by_subj = {m.subject_id: m for m in ms.mappings}
    # Only Organization/Organisation is borderline -> exactly one LLM call.
    assert provider.calls == 1
    org = by_subj["Organization"]
    assert org.matcher == "llm"
    assert org.predicate == MappingRelation.EXACT_MATCH
    assert org.needs_review is False  # LLM confidence 0.96 promoted it to auto-accept
    # High-confidence lexical matches never hit the LLM.
    assert by_subj["Person"].matcher == "lexical"


async def test_embedding_rescues_semantic_only_match() -> None:
    # "Firm" and "Company" are lexically dissimilar but embedded identically.
    src = GraphOntology(node_types=[NodeType(name="Firm")])
    tgt = GraphOntology(node_types=[NodeType(name="Company"), NodeType(name="Bicycle")])

    def embed(texts: Sequence[str]) -> list[list[float]]:
        table = {"Firm": [1.0, 0.0], "Company": [1.0, 0.0], "Bicycle": [0.0, 1.0]}
        return [table[t] for t in texts]

    aligner = OntologyAligner(
        LexicalMatcher(),
        embedding=EmbeddingMatcher(embed),
        thresholds=_THRESHOLDS,
    )
    ms = await aligner.align(src, tgt, source_id="s", target_id="t")
    firm = next(m for m in ms.mappings if m.subject_id == "Firm")
    assert firm.object_id == "Company"
    assert firm.matcher == "lexical+embedding"
    assert firm.needs_review is True  # combined score lands in the review band
