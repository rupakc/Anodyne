from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_graph.errors import GraphGenerationError
from anodyne_graph.generator import LLMGraphGenerator
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec


class _Provider(LLMProvider):
    def __init__(self, content: str) -> None:
        self._c = content

    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content=self._c, usage=Usage())

    async def _s(self) -> AsyncIterator[str]:
        if False:
            yield ""

    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        return self._s()


_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="gemini", model="g")

_ONTOLOGY = GraphOntology(
    node_types=[
        NodeType(
            name="Person",
            properties=[
                PropertySpec(name="name", datatype="string"),
                PropertySpec(name="age", datatype="integer"),
            ],
        ),
        NodeType(name="Company", properties=[PropertySpec(name="name", datatype="string")]),
    ],
    edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
)

# alice appears twice (dedup), Alien is unknown (dropped), bob's age is a string
# (coerced). Edges: valid alice->acme; bob->ghost dangling (drop); acme->alice
# type-mismatch (drop); KNOWS unknown edge type (drop).
_GRAPH_JSON = json.dumps(
    {
        "nodes": [
            {"type": "Person", "key": "alice", "properties": {"name": "REAL NAME", "age": 30}},
            {"type": "Person", "key": "alice", "properties": {"name": "dup", "age": 31}},
            {"type": "Person", "key": "bob", "properties": {"name": "Bob", "age": "40"}},
            {"type": "Company", "key": "acme", "properties": {"name": "Acme"}},
            {"type": "Alien", "key": "zorp", "properties": {}},
        ],
        "edges": [
            {"type": "WORKS_AT", "source": "alice", "target": "acme", "properties": {}},
            {"type": "WORKS_AT", "source": "bob", "target": "ghost", "properties": {}},
            {"type": "WORKS_AT", "source": "acme", "target": "alice", "properties": {}},
            {"type": "KNOWS", "source": "alice", "target": "bob", "properties": {}},
        ],
    }
)


def _spec() -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="g",
        description="people and companies",
        modality=Modality.GRAPH,
        source="description",
        fields=[],
        target_rows=10,
        directives={"ontology": _ONTOLOGY.model_dump(mode="json")},
    )


def test_dedup_and_unknown_type_dropped() -> None:
    ds = LLMGraphGenerator(_Provider(_GRAPH_JSON), _CFG).generate(_spec(), 0, 10, seed=1)
    ids = sorted(n.id for n in ds.nodes)
    assert ids == ["Company:acme", "Person:alice", "Person:bob"]
    assert ds.metrics["nodes_by_type"] == {"Company": 1, "Person": 2}


def test_referential_integrity_keeps_only_valid_edges() -> None:
    ds = LLMGraphGenerator(_Provider(_GRAPH_JSON), _CFG).generate(_spec(), 0, 10, seed=1)
    assert len(ds.edges) == 1
    e = ds.edges[0]
    assert e.type == "WORKS_AT" and e.source == "Person:alice" and e.target == "Company:acme"


def test_pii_property_is_faked_not_llm_value() -> None:
    ds = LLMGraphGenerator(_Provider(_GRAPH_JSON), _CFG).generate(_spec(), 0, 10, seed=1)
    alice = next(n for n in ds.nodes if n.id == "Person:alice")
    # `name` is PII -> always faked, never the LLM-provided literal.
    assert alice.properties["name"] != "REAL NAME"
    assert isinstance(alice.properties["name"], str) and alice.properties["name"]


def test_non_pii_value_coerced_to_datatype() -> None:
    ds = LLMGraphGenerator(_Provider(_GRAPH_JSON), _CFG).generate(_spec(), 0, 10, seed=1)
    bob = next(n for n in ds.nodes if n.id == "Person:bob")
    assert bob.properties["age"] == 40  # "40" (string) coerced to int


def test_same_seed_is_deterministic() -> None:
    from anodyne_graph.serialization import to_json_bytes

    a = LLMGraphGenerator(_Provider(_GRAPH_JSON), _CFG).generate(_spec(), 0, 10, seed=7)
    b = LLMGraphGenerator(_Provider(_GRAPH_JSON), _CFG).generate(_spec(), 0, 10, seed=7)
    assert to_json_bytes(a) == to_json_bytes(b)


def test_raises_without_ontology() -> None:
    spec = _spec()
    spec.directives = {}
    with pytest.raises(GraphGenerationError):
        LLMGraphGenerator(_Provider(_GRAPH_JSON), _CFG).generate(spec, 0, 10, seed=1)


def test_raises_when_no_valid_nodes() -> None:
    empty = json.dumps({"nodes": [{"type": "Alien", "key": "z"}], "edges": []})
    with pytest.raises(GraphGenerationError):
        LLMGraphGenerator(_Provider(empty), _CFG).generate(_spec(), 0, 10, seed=1)
