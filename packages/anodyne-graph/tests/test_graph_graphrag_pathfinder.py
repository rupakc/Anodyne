from __future__ import annotations

import pytest
from anodyne_graph.errors import GraphRAGError
from anodyne_graph.graphrag.models import QAPath
from anodyne_graph.graphrag.pathfinder import sample_paths
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
)


def _dataset() -> GraphDataset:
    ontology = GraphOntology(
        node_types=[NodeType(name="Person"), NodeType(name="Company")],
        edge_types=[
            EdgeType(name="KNOWS", source_type="Person", target_type="Person"),
            EdgeType(name="WORKS_AT", source_type="Person", target_type="Company"),
        ],
    )
    nodes = [Node(id=f"Person:{i}", type="Person") for i in range(5)] + [
        Node(id=f"Company:{i}", type="Company") for i in range(2)
    ]
    edges = [
        Edge(id="K0", type="KNOWS", source="Person:0", target="Person:1"),
        Edge(id="K1", type="KNOWS", source="Person:1", target="Person:2"),
        Edge(id="K2", type="KNOWS", source="Person:2", target="Person:3"),
        Edge(id="K3", type="KNOWS", source="Person:0", target="Person:4"),
        Edge(id="W0", type="WORKS_AT", source="Person:3", target="Company:0"),
        Edge(id="W1", type="WORKS_AT", source="Person:1", target="Company:1"),
    ]
    return GraphDataset(ontology=ontology, nodes=nodes, edges=edges)


def _assert_path_grounded(path: QAPath, dataset: GraphDataset) -> None:
    edges_by_id = {e.id: e for e in dataset.edges}
    directed = {et.name: et.directed for et in dataset.ontology.edge_types}
    node_ids = path.node_ids
    assert len(node_ids) == len(set(node_ids))  # simple path, no revisited nodes
    for i, (node_id, edge_id) in enumerate(path.hops):
        edge = edges_by_id[edge_id]
        nxt = node_ids[i + 1]
        if directed.get(edge.type, True):
            assert (edge.source, edge.target) == (node_id, nxt)
        else:
            assert {edge.source, edge.target} == {node_id, nxt}


def test_sampled_paths_are_valid_connected_multi_hop() -> None:
    ds = _dataset()
    paths = sample_paths(ds, count=5, seed=7, min_hops=2, max_hops=4)
    assert paths
    for path in paths:
        assert 2 <= path.hop_count <= 4
        _assert_path_grounded(path, ds)


def test_sampling_is_deterministic_for_same_seed() -> None:
    ds = _dataset()
    a = sample_paths(ds, count=4, seed=42, min_hops=2, max_hops=3)
    b = sample_paths(ds, count=4, seed=42, min_hops=2, max_hops=3)
    assert [p.model_dump() for p in a] == [p.model_dump() for p in b]


def test_paths_are_distinct_and_count_capped() -> None:
    ds = _dataset()
    paths = sample_paths(ds, count=3, seed=1, min_hops=2, max_hops=4)
    assert len(paths) <= 3
    keys = {(tuple(p.hops), p.terminal_node_id) for p in paths}
    assert len(keys) == len(paths)


def test_empty_graph_raises() -> None:
    empty = GraphDataset(ontology=GraphOntology())
    with pytest.raises(GraphRAGError):
        sample_paths(empty, count=1, seed=0)


def test_too_small_graph_raises() -> None:
    ontology = GraphOntology(
        node_types=[NodeType(name="Person")],
        edge_types=[EdgeType(name="KNOWS", source_type="Person", target_type="Person")],
    )
    tiny = GraphDataset(
        ontology=ontology,
        nodes=[Node(id="Person:0", type="Person"), Node(id="Person:1", type="Person")],
        edges=[Edge(id="K0", type="KNOWS", source="Person:0", target="Person:1")],
    )
    with pytest.raises(GraphRAGError):
        sample_paths(tiny, count=1, seed=0, min_hops=2, max_hops=4)
