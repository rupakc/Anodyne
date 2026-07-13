from __future__ import annotations

import pytest
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
from anodyne_graph.serialization import from_json_bytes, to_json_bytes


def _dataset() -> GraphDataset:
    nodes = [
        Node(id="Person:alice", type="Person", properties={"name": "Alice", "age": 30}),
        Node(id="Company:acme", type="Company", properties={"name": "Acme"}),
    ]
    edges = [Edge(id="WORKS_AT:0", type="WORKS_AT", source="Person:alice", target="Company:acme")]
    ontology = GraphOntology(
        node_types=[
            NodeType(name="Person", properties=[PropertySpec(name="age", datatype="integer")]),
            NodeType(name="Company"),
        ],
        edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
    )
    return GraphDataset(
        ontology=ontology, nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def test_round_trip_is_lossless() -> None:
    ds = _dataset()
    restored = from_json_bytes(to_json_bytes(ds))
    assert restored == ds


def test_serialization_is_deterministic_bytes() -> None:
    ds = _dataset()
    assert to_json_bytes(ds) == to_json_bytes(ds)


def test_from_json_bytes_rejects_garbage() -> None:
    with pytest.raises(ValueError):
        from_json_bytes(b"not json")


def test_from_json_bytes_carries_node_link_shape() -> None:
    payload = to_json_bytes(_dataset())
    restored = from_json_bytes(payload)
    assert {n.id for n in restored.nodes} == {"Person:alice", "Company:acme"}
    assert restored.edges[0].source == "Person:alice"
    assert restored.metrics["node_count"] == 2
