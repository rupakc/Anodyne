from __future__ import annotations

import json
from collections.abc import AsyncIterator
from uuid import uuid4

from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_graph.hybrid import HybridGraphGenerator
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec
from anodyne_graph.serialization import to_json_bytes


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


_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="gemini", model="g")

_ONTOLOGY = GraphOntology(
    node_types=[
        NodeType(
            name="Person",
            properties=[
                PropertySpec(name="full_name", datatype="string"),  # PII -> faked, never LLM
                PropertySpec(name="occupation", datatype="string"),  # non-PII -> LLM-filled
                PropertySpec(name="age", datatype="integer"),
            ],
        )
    ],
    edge_types=[EdgeType(name="KNOWS", source_type="Person", target_type="Person")],
)

# LLM returns realistic non-PII values (no names). One row -> cycled across nodes.
_FILL = json.dumps({"Person": [{"occupation": "Botanist", "age": 42}]})


def _spec() -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="g",
        description="a social network of people",
        modality=Modality.GRAPH,
        source="description",
        fields=[],
        target_rows=30,
        directives={
            "method": "hybrid",
            "topology": "barabasi_albert",
            "m": 2,
            "ontology": _ONTOLOGY.model_dump(mode="json"),
        },
    )


def test_hybrid_fills_non_pii_from_llm_and_fakes_pii() -> None:
    gen = HybridGraphGenerator(_Provider(_FILL), _CFG)
    ds = gen.generate(_spec(), 0, 30, seed=1)
    assert ds.metrics["engine"] == "hybrid"
    for n in ds.nodes:
        assert n.properties["occupation"] == "Botanist"  # from the LLM
        assert n.properties["age"] == 42
        assert n.properties["full_name"] != "Botanist"  # PII faked, not the LLM value
        assert isinstance(n.properties["full_name"], str) and n.properties["full_name"]


def test_hybrid_topology_preserved() -> None:
    gen = HybridGraphGenerator(_Provider(_FILL), _CFG)
    ds = gen.generate(_spec(), 0, 30, seed=1)
    assert len(ds.nodes) == 30
    assert len(ds.edges) > 0  # BA structure survives the semantic fill


def test_hybrid_deterministic_with_mocked_llm() -> None:
    a = HybridGraphGenerator(_Provider(_FILL), _CFG).generate(_spec(), 0, 30, seed=9)
    b = HybridGraphGenerator(_Provider(_FILL), _CFG).generate(_spec(), 0, 30, seed=9)
    assert to_json_bytes(a) == to_json_bytes(b)


def test_hybrid_tolerates_bad_llm_output() -> None:
    ds = HybridGraphGenerator(_Provider("not json"), _CFG).generate(_spec(), 0, 20, seed=2)
    # falls back to the topology-synthesized values; still a valid graph
    assert len(ds.nodes) == 20
    for n in ds.nodes:
        assert "occupation" in n.properties
