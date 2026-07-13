from __future__ import annotations

import io
import zipfile

import networkx as nx  # type: ignore[import-untyped]
from anodyne_graph.export import encode_dataset
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
from anodyne_graph.serialization import from_json_bytes


def _dataset() -> GraphDataset:
    nodes = [
        Node(id="p1", type="Person", properties={"name": "Alice", "age": 30}),
        Node(id="p2", type="Person", properties={"name": "Bob", "age": 25}),
        Node(id="c1", type="Company", properties={"name": "Acme"}),
    ]
    edges = [
        Edge(id="e1", type="WORKS_AT", source="p1", target="c1", properties={"since": 2020}),
        Edge(id="e2", type="WORKS_AT", source="p2", target="c1", properties={"since": 2021}),
        Edge(id="e3", type="KNOWS", source="p1", target="p2"),
    ]
    ontology = GraphOntology(
        node_types=[
            NodeType(name="Person", properties=[PropertySpec(name="age", datatype="integer")]),
            NodeType(name="Company"),
        ],
        edge_types=[
            EdgeType(name="WORKS_AT", source_type="Person", target_type="Company"),
            EdgeType(name="KNOWS", source_type="Person", target_type="Person"),
        ],
    )
    return GraphDataset(
        ontology=ontology, nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def test_graphml_round_trips_via_networkx() -> None:
    data = encode_dataset(_dataset(), "graphml")
    g = nx.read_graphml(io.BytesIO(data))
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert g.nodes["p1"]["type"] == "Person"


def test_gexf_round_trips_via_networkx() -> None:
    data = encode_dataset(_dataset(), "gexf")
    g = nx.read_gexf(io.BytesIO(data))
    assert g.number_of_nodes() == 3
    assert g.number_of_edges() == 3
    assert g.nodes["c1"]["type"] == "Company"


def test_cypher_contains_expected_create_lines() -> None:
    data = encode_dataset(_dataset(), "cypher").decode("utf-8")
    lines = data.splitlines()
    node_creates = [line for line in lines if line.startswith("CREATE (:")]
    edge_creates = [line for line in lines if "CREATE (a)-[:" in line]
    assert len(node_creates) == 3
    assert len(edge_creates) == 3
    assert any(
        line.startswith("CREATE (:Person ") and '_nid: "p1"' in line for line in node_creates
    )
    # Endpoints are matched on the same synthetic `_nid` key they were created with.
    assert any('MATCH (a {_nid: "p1"}), (b {_nid: "c1"})' in line for line in lines)
    assert any("[:WORKS_AT" in line for line in edge_creates)


def test_cypher_output_is_deterministic() -> None:
    assert encode_dataset(_dataset(), "cypher") == encode_dataset(_dataset(), "cypher")


def test_neo4j_csv_zip_has_expected_headers_and_rows() -> None:
    data = encode_dataset(_dataset(), "neo4j-csv")
    zf = zipfile.ZipFile(io.BytesIO(data))
    names = set(zf.namelist())
    assert "nodes_Person.csv" in names
    assert "nodes_Company.csv" in names
    assert "edges_WORKS_AT.csv" in names
    assert "edges_KNOWS.csv" in names

    person_csv = zf.read("nodes_Person.csv").decode("utf-8").splitlines()
    assert person_csv[0] == "id:ID,:LABEL,age,name"
    assert len(person_csv) == 3  # header + 2 Person rows

    works_at_csv = zf.read("edges_WORKS_AT.csv").decode("utf-8").splitlines()
    assert works_at_csv[0] == ":START_ID,:END_ID,:TYPE,since"
    assert len(works_at_csv) == 3  # header + 2 WORKS_AT rows


def _id_collision_dataset() -> GraphDataset:
    """An ontology whose properties literally collide with reserved/lookup keys
    (`id`, `type`, `key`, `directed`) -- the regression fixture for both the
    networkx reserved-kwarg crash and the Cypher `_nid` matching bug."""
    nodes = [
        Node(
            id="a",
            type="Thing",
            properties={"id": "USER-1", "type": "x", "key": "k", "directed": "no"},
        ),
        Node(
            id="b",
            type="Thing",
            properties={"id": "USER-2", "type": "y", "key": "j", "directed": "yes"},
        ),
    ]
    edges = [Edge(id="r1", type="LINKS", source="a", target="b", properties={"id": "EDGE-9"})]
    ontology = GraphOntology(
        node_types=[
            NodeType(
                name="Thing",
                properties=[
                    PropertySpec(name="id", datatype="string"),
                    PropertySpec(name="type", datatype="string"),
                    PropertySpec(name="key", datatype="string"),
                    PropertySpec(name="directed", datatype="string"),
                ],
            )
        ],
        edge_types=[EdgeType(name="LINKS", source_type="Thing", target_type="Thing")],
    )
    return GraphDataset(
        ontology=ontology, nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


def test_graphml_gexf_export_with_reserved_kwarg_property_names() -> None:
    # A property named `type`/`id`/`key`/`directed` must not crash the writer
    # (networkx reserved-kwarg TypeError regression).
    ds = _id_collision_dataset()
    gml = nx.read_graphml(io.BytesIO(encode_dataset(ds, "graphml")))
    assert gml.number_of_nodes() == 2 and gml.number_of_edges() == 1
    assert gml.nodes["a"]["type"] == "Thing"  # structural type wins over the property
    gexf = nx.read_gexf(io.BytesIO(encode_dataset(ds, "gexf")))
    assert gexf.number_of_nodes() == 2 and gexf.number_of_edges() == 1


def test_cypher_with_id_property_still_creates_all_relationships() -> None:
    # A user property named `id` must not clobber the endpoint lookup key.
    data = encode_dataset(_id_collision_dataset(), "cypher").decode("utf-8")
    lines = data.splitlines()
    edge_creates = [line for line in lines if "CREATE (a)-[:" in line]
    assert len(edge_creates) == 1  # the relationship is created, not silently dropped
    # The MATCH uses `_nid` (the real node id), not the user `id` property.
    assert 'MATCH (a {_nid: "a"}), (b {_nid: "b"})' in data
    # The user `id` property is preserved verbatim on the created nodes.
    assert 'id: "USER-1"' in data and 'id: "USER-2"' in data


def test_graph_json_is_passthrough_and_lossless() -> None:
    data = encode_dataset(_dataset(), "graph_json")
    restored = from_json_bytes(data)
    assert restored == _dataset()
