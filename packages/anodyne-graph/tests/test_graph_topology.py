from __future__ import annotations

from uuid import uuid4

import networkx as nx  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_graph.errors import GraphGenerationError
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec
from anodyne_graph.serialization import to_json_bytes
from anodyne_graph.topology import ProceduralTopologyGenerator

# Homogeneous ontology: one node type + a self-relation, so every topology edge
# maps (no drops) and the raw structural statistics are preserved exactly.
_SELF = GraphOntology(
    node_types=[
        NodeType(
            name="Person",
            properties=[
                PropertySpec(name="full_name", datatype="string"),  # PII -> faked
                PropertySpec(name="age", datatype="integer"),
                PropertySpec(name="tier", datatype="string", constraints={"choices": ["a", "b"]}),
            ],
        )
    ],
    edge_types=[EdgeType(name="KNOWS", source_type="Person", target_type="Person")],
)

# Two node types so edge mapping must respect domain/range (and drop the rest).
_BIPARTITE = GraphOntology(
    node_types=[
        NodeType(name="Person", properties=[PropertySpec(name="age", datatype="integer")]),
        NodeType(name="Company", properties=[]),
    ],
    edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
)


def _spec(topology: str, ontology: GraphOntology, **params: object) -> DatasetSpec:
    directives: dict[str, object] = {
        "topology": topology,
        "ontology": ontology.model_dump(mode="json"),
    }
    directives.update(params)
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="g",
        description="graph",
        modality=Modality.GRAPH,
        source="description",
        fields=[],
        target_rows=100,
        directives=directives,
    )


def _nx_from(ds: object) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(n.id for n in ds.nodes)  # type: ignore[attr-defined]
    g.add_edges_from((e.source, e.target) for e in ds.edges)  # type: ignore[attr-defined]
    return g


def test_barabasi_albert_is_scale_free() -> None:
    ds = ProceduralTopologyGenerator().generate(
        _spec("barabasi_albert", _SELF, m=2), 0, 300, seed=1
    )
    g = _nx_from(ds)
    degrees = [d for _, d in g.degree()]
    mean = sum(degrees) / len(degrees)
    # scale-free: a heavy tail -> the max degree dwarfs the mean.
    assert max(degrees) > 4 * mean
    assert ds.metrics["topology"] == "barabasi_albert"
    assert ds.metrics["edges_dropped_unmappable"] == 0


def test_watts_strogatz_and_erdos_renyi_produce_mapped_edges() -> None:
    ws = ProceduralTopologyGenerator().generate(
        _spec("watts_strogatz", _SELF, k=4, p=0.2), 0, 80, seed=2
    )
    er = ProceduralTopologyGenerator().generate(_spec("erdos_renyi", _SELF, p=0.1), 0, 80, seed=2)
    assert len(ws.nodes) == 80 and len(ws.edges) > 0
    assert len(er.nodes) == 80 and len(er.edges) > 0


def test_stochastic_block_model_has_community_structure() -> None:
    ds = ProceduralTopologyGenerator().generate(
        _spec("stochastic_block_model", _SELF, n_blocks=3, p_in=0.35, p_out=0.01), 0, 90, seed=3
    )
    assert ds.metrics["community_count"] == 3
    g = _nx_from(ds)
    from networkx.algorithms.community import (  # type: ignore[import-untyped]
        greedy_modularity_communities,
        modularity,
    )

    communities = greedy_modularity_communities(g)
    assert modularity(g, communities) > 0.3


def test_lfr_produces_communities() -> None:
    ds = ProceduralTopologyGenerator().generate(
        _spec("lfr", _SELF, mu=0.1, average_degree=5, min_community=20), 0, 250, seed=10
    )
    assert ds.metrics["community_count"] >= 2
    assert len(ds.edges) > 0


def test_bipartite_ontology_maps_domain_range_and_counts_drops() -> None:
    ds = ProceduralTopologyGenerator().generate(
        _spec("erdos_renyi", _BIPARTITE, p=0.1), 0, 60, seed=4
    )
    for e in ds.edges:
        src = next(n for n in ds.nodes if n.id == e.source)
        tgt = next(n for n in ds.nodes if n.id == e.target)
        assert src.type == "Person" and tgt.type == "Company"
    # Person-Person / Company-Company topology edges have no compatible edge type.
    assert ds.metrics["edges_dropped_unmappable"] > 0


def test_pii_faked_and_choices_respected() -> None:
    ds = ProceduralTopologyGenerator().generate(_spec("barabasi_albert", _SELF, m=2), 0, 40, seed=5)
    for n in ds.nodes:
        assert isinstance(n.properties["full_name"], str) and n.properties["full_name"]
        assert n.properties["tier"] in ("a", "b")


def test_same_seed_is_deterministic() -> None:
    a = ProceduralTopologyGenerator().generate(_spec("barabasi_albert", _SELF, m=3), 0, 120, seed=7)
    b = ProceduralTopologyGenerator().generate(_spec("barabasi_albert", _SELF, m=3), 0, 120, seed=7)
    assert to_json_bytes(a) == to_json_bytes(b)


def test_unknown_topology_raises() -> None:
    with pytest.raises(GraphGenerationError):
        ProceduralTopologyGenerator().generate(_spec("nonsense", _SELF), 0, 10, seed=1)
