from __future__ import annotations

from collections.abc import AsyncIterator
from uuid import uuid4

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_core.ports import LLMProvider
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_graph.engines import (
    build_graph_engine,
    generate_shard,
    is_from_sample,
    needs_llm,
)
from anodyne_graph.errors import GraphGenerationError
from anodyne_graph.from_sample import FromSampleGraphGenerator
from anodyne_graph.generator import LLMGraphGenerator
from anodyne_graph.hybrid import HybridGraphGenerator
from anodyne_graph.models import (
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
)
from anodyne_graph.topology import ProceduralTopologyGenerator

_CFG = ModelConfig(id=uuid4(), tenant_id=uuid4(), name="m", provider="gemini", model="g")

_ONT = GraphOntology(
    node_types=[NodeType(name="Person", properties=[PropertySpec(name="age", datatype="integer")])],
    edge_types=[EdgeType(name="KNOWS", source_type="Person", target_type="Person")],
)


class _Provider(LLMProvider):
    async def complete(self, config: ModelConfig, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="{}", usage=Usage())

    async def _s(self) -> AsyncIterator[str]:
        if False:
            yield ""

    def stream(self, config: ModelConfig, request: LLMRequest) -> AsyncIterator[str]:
        return self._s()


def _spec(source: str = "description", **directives: object) -> DatasetSpec:
    d: dict[str, object] = {"ontology": _ONT.model_dump(mode="json")}
    d.update(directives)
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="g",
        description="people",
        modality=Modality.GRAPH,
        source=source,
        fields=[],
        target_rows=20,
        directives=d,
    )


def test_selection_flags() -> None:
    assert is_from_sample(_spec(source="sample"))
    assert is_from_sample(_spec(method="from_sample"))
    assert needs_llm(_spec())  # default LLM path
    assert needs_llm(_spec(method="hybrid"))
    assert not needs_llm(_spec(topology="barabasi_albert"))
    assert not needs_llm(_spec(source="sample"))


def test_build_engine_dispatch() -> None:
    assert isinstance(build_graph_engine(_spec(), _Provider(), _CFG), LLMGraphGenerator)
    assert isinstance(
        build_graph_engine(_spec(topology="erdos_renyi"), None, None), ProceduralTopologyGenerator
    )
    assert isinstance(
        build_graph_engine(_spec(method="hybrid"), _Provider(), _CFG), HybridGraphGenerator
    )
    sample = GraphDataset(
        ontology=_ONT,
        nodes=[Node(id="Person:0", type="Person", properties={"age": 1})],
        edges=[],
    )
    assert isinstance(
        build_graph_engine(_spec(source="sample"), None, None, sample=sample),
        FromSampleGraphGenerator,
    )


def test_from_sample_without_sample_raises() -> None:
    with pytest.raises(GraphGenerationError):
        build_graph_engine(_spec(source="sample"), None, None)


def test_generate_shard_topology_with_constraint_layer() -> None:
    ds = generate_shard(
        _spec(topology="erdos_renyi", p=0.2, validate=True), None, None, 0, 30, seed=1
    )
    assert ds.metrics["engine"] == "topology"
    assert ds.metrics["constraint_conforms"] is True
    assert ds.metrics["constraint_violations"] == 0


def test_generate_shard_injects_and_reports_violations() -> None:
    ds = generate_shard(
        _spec(topology="erdos_renyi", p=0.2, inject_violations=3), None, None, 0, 30, seed=1
    )
    assert ds.metrics["injected_violations"] == 3
    assert ds.metrics["constraint_violations"] >= 3
    assert ds.metrics["constraint_conforms"] is False
