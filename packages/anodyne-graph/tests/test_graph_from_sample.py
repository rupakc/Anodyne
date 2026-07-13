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


_PRIVACY_ONTOLOGY = GraphOntology(
    node_types=[
        NodeType(
            name="Employee",
            properties=[
                # High-cardinality free text: must be synthesized, never copied.
                PropertySpec(name="notes", datatype="string"),
                # Sensitive numeric: must be sampled, never a verbatim copy.
                PropertySpec(name="salary", datatype="float"),
                # Low-cardinality short categorical: resampling labels is allowed.
                PropertySpec(name="department", datatype="string"),
            ],
        ),
    ],
    edge_types=[EdgeType(name="KNOWS", source_type="Employee", target_type="Employee")],
)


def _privacy_sample() -> GraphDataset:
    nodes: list[Node] = []
    edges: list[Edge] = []
    depts = ["Sales", "Eng", "Legal"]
    for i in range(30):
        eid = f"Employee:{i}"
        nodes.append(
            Node(
                id=eid,
                type="Employee",
                properties={
                    # Every note is a distinct, long, unique free-text string.
                    "notes": f"Confidential performance memo #{i} — do not share {i * 7919}",
                    "salary": 50_000.0 + i * 1_234.5,
                    "department": depts[i % len(depts)],
                },
            )
        )
        if i:
            edges.append(
                Edge(id=f"KNOWS:{i}", type="KNOWS", source=eid, target=f"Employee:{i - 1}")
            )
    return GraphDataset(ontology=_PRIVACY_ONTOLOGY, nodes=nodes, edges=edges)


def test_freetext_synthesized_not_copied_and_numeric_not_verbatim() -> None:
    sample = _privacy_sample()
    ds = FromSampleGraphGenerator(sample).generate(_spec(), 0, 40, seed=3)

    sample_notes = {str(n.properties["notes"]) for n in sample.nodes}
    synth_notes = {str(n.properties["notes"]) for n in ds.nodes}
    # High-cardinality free text is synthesized -> disjoint from the sample.
    assert synth_notes.isdisjoint(sample_notes)

    sample_salaries = {float(n.properties["salary"]) for n in sample.nodes}
    synth_salaries = [float(n.properties["salary"]) for n in ds.nodes]
    lo, hi = min(sample_salaries), max(sample_salaries)
    # Numeric values are sampled from a fitted range: in-range but not verbatim.
    assert all(lo <= s <= hi for s in synth_salaries)
    assert any(s not in sample_salaries for s in synth_salaries)

    # Low-cardinality department labels may recur (statistical matching, by design).
    synth_depts = {str(n.properties["department"]) for n in ds.nodes}
    assert synth_depts <= {"Sales", "Eng", "Legal"}


def test_shards_produce_disjoint_node_ids() -> None:
    sample = _sample()
    gen = FromSampleGraphGenerator(sample)
    shard0 = gen.generate(_spec(), 0, 20, seed=1, shard_index=0)
    shard1 = gen.generate(_spec(), 20, 20, seed=1, shard_index=1)
    ids0 = {n.id for n in shard0.nodes}
    ids1 = {n.id for n in shard1.nodes}
    assert ids0.isdisjoint(ids1)


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
