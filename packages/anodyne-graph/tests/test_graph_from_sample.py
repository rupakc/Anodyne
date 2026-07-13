from __future__ import annotations

from uuid import uuid4

import networkx as nx  # type: ignore[import-untyped]
from anodyne_dataset.models import DatasetSpec, Modality
from anodyne_graph.from_sample import (
    FromSampleGraphGenerator,
    assert_no_verbatim_subgraph,
    graphml_to_dataset,
)
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
)
from anodyne_graph.serialization import to_json_bytes

_ONTOLOGY = GraphOntology(
    node_types=[
        NodeType(
            name="Person",
            properties=[
                PropertySpec(name="full_name", datatype="string"),  # PII
                PropertySpec(name="age", datatype="integer"),
            ],
        ),
        NodeType(name="Company", properties=[PropertySpec(name="full_name", datatype="string")]),
    ],
    edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
)


def _sample() -> GraphDataset:
    nodes: list[Node] = []
    edges: list[Edge] = []
    companies = [f"Company:{i}" for i in range(4)]
    for c in companies:
        nodes.append(Node(id=c, type="Company", properties={"full_name": f"Corp {c}"}))
    # power-law-ish: company 0 hires many, others few
    for i in range(24):
        pid = f"Person:{i}"
        nodes.append(
            Node(id=pid, type="Person", properties={"full_name": f"Real Person {i}", "age": 20 + i})
        )
        target = companies[0] if i % 3 else companies[i % 4]
        edges.append(Edge(id=f"WORKS_AT:{i}", type="WORKS_AT", source=pid, target=target))
    return GraphDataset(ontology=_ONTOLOGY, nodes=nodes, edges=edges)


def _spec() -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="g",
        description="employment graph",
        modality=Modality.GRAPH,
        source="sample",
        fields=[],
        target_rows=40,
        directives={},
    )


def test_matches_degree_stats_within_tolerance() -> None:
    sample = _sample()
    gen = FromSampleGraphGenerator(sample)
    ds = gen.generate(_spec(), 0, 40, seed=1)
    assert len(ds.nodes) == 40
    g = nx.Graph()
    g.add_nodes_from(n.id for n in ds.nodes)
    g.add_edges_from((e.source, e.target) for e in ds.edges)
    degrees = [d for _, d in g.degree()]
    synth_mean = sum(degrees) / len(degrees)
    sample_mean = float(ds.metrics["sample_mean_degree"])
    assert abs(synth_mean - sample_mean) < 0.5 * sample_mean + 2.0


def test_no_verbatim_subgraph_guarantee() -> None:
    sample = _sample()
    ds = FromSampleGraphGenerator(sample).generate(_spec(), 0, 40, seed=1)
    assert ds.metrics["verbatim_subgraph_overlap"] == 0
    # And directly: fingerprints are disjoint (PII re-faked, ids fresh).
    assert assert_no_verbatim_subgraph(sample, ds, strict=True) == 0


def test_pii_is_refaked_not_copied() -> None:
    sample = _sample()
    ds = FromSampleGraphGenerator(sample).generate(_spec(), 0, 40, seed=1)
    original_names = {n.properties["full_name"] for n in sample.nodes}
    for n in ds.nodes:
        assert n.properties["full_name"] not in original_names


def test_ages_within_sample_range() -> None:
    sample = _sample()
    ds = FromSampleGraphGenerator(sample).generate(_spec(), 0, 40, seed=2)
    for n in ds.nodes:
        if n.type == "Person":
            assert 20 <= n.properties["age"] <= 43


def test_deterministic_same_seed() -> None:
    sample = _sample()
    a = FromSampleGraphGenerator(sample).generate(_spec(), 0, 40, seed=5)
    b = FromSampleGraphGenerator(sample).generate(_spec(), 0, 40, seed=5)
    assert to_json_bytes(a) == to_json_bytes(b)


def test_graphml_round_trip_parse() -> None:
    g = nx.DiGraph()
    g.add_node("a", type="Person", age="30")
    g.add_node("b", type="Company")
    g.add_edge("a", "b", type="WORKS_AT")
    import io

    buf = io.BytesIO()
    nx.write_graphml(g, buf)
    ds = graphml_to_dataset(buf.getvalue())
    assert {nt.name for nt in ds.ontology.node_types} == {"Person", "Company"}
    assert any(e.type == "WORKS_AT" for e in ds.edges)
    # and it can drive synthesis
    gen = FromSampleGraphGenerator.from_graphml_bytes(buf.getvalue())
    out = gen.generate(_spec(), 0, 6, seed=1)
    assert len(out.nodes) == 6
