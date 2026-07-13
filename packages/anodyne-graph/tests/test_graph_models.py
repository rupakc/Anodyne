from __future__ import annotations

from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
    compute_metrics,
)


def _ontology() -> GraphOntology:
    return GraphOntology(
        node_types=[
            NodeType(name="Person", properties=[PropertySpec(name="age", datatype="integer")]),
            NodeType(name="Company"),
        ],
        edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
    )


def test_property_spec_defaults() -> None:
    p = PropertySpec(name="x")
    assert p.datatype == "string" and p.nullable is False and p.constraints == {}


def test_edge_type_defaults_directed() -> None:
    et = EdgeType(name="R", source_type="A", target_type="B")
    assert et.directed is True


def test_ontology_lookup_helpers() -> None:
    onto = _ontology()
    assert onto.node_type("Person") is not None
    assert onto.node_type("Nope") is None
    works_at = onto.edge_type("WORKS_AT")
    assert works_at is not None
    assert works_at.source_type == "Person"


def test_ontology_has_subclass_of_seam() -> None:
    assert GraphOntology().subclass_of == {}


def test_compute_metrics_counts_total_and_per_type() -> None:
    nodes = [
        Node(id="Person:1", type="Person"),
        Node(id="Person:2", type="Person"),
        Node(id="Company:1", type="Company"),
    ]
    edges = [Edge(id="e0", type="WORKS_AT", source="Person:1", target="Company:1")]
    metrics = compute_metrics(nodes, edges)
    assert metrics["node_count"] == 3
    assert metrics["edge_count"] == 1
    assert metrics["nodes_by_type"] == {"Company": 1, "Person": 2}
    assert metrics["edges_by_type"] == {"WORKS_AT": 1}


def test_graph_dataset_holds_ontology_and_instances() -> None:
    ds = GraphDataset(ontology=_ontology(), nodes=[Node(id="Person:1", type="Person")])
    assert ds.nodes[0].id == "Person:1"
    assert ds.edges == []
