from __future__ import annotations

import io

import numpy as np
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from anodyne_graph.export import encode_dataset
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    compute_metrics,
)


def _dataset() -> GraphDataset:
    nodes = [
        Node(id="p1", type="Person"),
        Node(id="p2", type="Person"),
        Node(id="c1", type="Company"),
    ]
    edges = [
        Edge(id="e1", type="WORKS_AT", source="p1", target="c1"),
        Edge(id="e2", type="WORKS_AT", source="p2", target="c1"),
        Edge(id="e3", type="KNOWS", source="p1", target="p2"),
    ]
    ontology = GraphOntology(
        node_types=[NodeType(name="Person"), NodeType(name="Company")],
        edge_types=[
            EdgeType(name="WORKS_AT", source_type="Person", target_type="Company"),
            EdgeType(name="KNOWS", source_type="Person", target_type="Person"),
        ],
    )
    return GraphDataset(
        ontology=ontology, nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def test_npz_edge_index_and_features_shape() -> None:
    data = encode_dataset(_dataset(), "npz")
    npz = np.load(io.BytesIO(data), allow_pickle=True)

    node_ids = list(npz["node_ids"])
    assert node_ids == ["p1", "p2", "c1"]

    edge_index = npz["edge_index"]
    assert edge_index.shape == (2, 3)
    # e1: p1(0) -> c1(2)
    assert edge_index[0, 0] == 0 and edge_index[1, 0] == 2

    node_features = npz["node_features"]
    assert node_features.shape == (3, 2)  # 3 nodes, 2 node types (one-hot)
    assert node_features.sum() == 3  # exactly one hot entry per node

    node_type_names = list(npz["node_type_names"])
    assert node_type_names == sorted({"Person", "Company"})


def test_npz_is_deterministic() -> None:
    assert encode_dataset(_dataset(), "npz") == encode_dataset(_dataset(), "npz")


def test_npz_drops_dangling_edges_without_raising() -> None:
    ds = _dataset()
    ds.edges.append(Edge(id="bad", type="WORKS_AT", source="p1", target="does-not-exist"))
    data = encode_dataset(ds, "npz")
    npz = np.load(io.BytesIO(data), allow_pickle=True)
    assert npz["edge_index"].shape == (2, 3)  # dangling edge dropped, not raised


def test_graph_parquet_round_trips_via_pyarrow() -> None:
    data = encode_dataset(_dataset(), "graph-parquet")
    table = pq.read_table(io.BytesIO(data))
    assert table.num_rows == 3
    assert set(table.column_names) == {"edge_id", "source", "target", "type", "directed"}
    assert sorted(table.column("edge_id").to_pylist()) == ["e1", "e2", "e3"]
