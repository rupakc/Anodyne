from __future__ import annotations

from anodyne_graph.export import EX, ONTO, dataset_to_rdf, encode_dataset
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
from rdflib import Graph


def _dataset() -> GraphDataset:
    nodes = [
        Node(id="p1", type="Person", properties={"name": "Alice", "age": 30}),
        Node(id="c1", type="Company", properties={"name": "Acme"}),
    ]
    edges = [
        Edge(
            id="e1",
            type="WORKS_AT",
            source="p1",
            target="c1",
            properties={"since": 2020, "role": "engineer"},
        )
    ]
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


def test_dataset_to_rdf_has_node_type_and_property_triples() -> None:
    from rdflib.namespace import RDF

    g = dataset_to_rdf(_dataset())
    assert (EX["p1"], RDF.type, ONTO["Person"]) in g
    assert (EX["c1"], RDF.type, ONTO["Company"]) in g
    assert (EX["p1"], ONTO["name"], None) in g
    assert (EX["p1"], ONTO["age"], None) in g


def test_dataset_to_rdf_edge_is_a_direct_triple() -> None:
    g = dataset_to_rdf(_dataset())
    assert (EX["p1"], ONTO["WORKS_AT"], EX["c1"]) in g


def test_dataset_to_rdf_edge_properties_use_reification() -> None:
    from rdflib.namespace import RDF

    g = dataset_to_rdf(_dataset())
    statements = list(g.subjects(RDF.type, RDF.Statement))
    assert len(statements) == 1
    stmt = statements[0]
    assert (stmt, RDF.subject, EX["p1"]) in g
    assert (stmt, RDF.predicate, ONTO["WORKS_AT"]) in g
    assert (stmt, RDF.object, EX["c1"]) in g
    assert (stmt, ONTO["since"], None) in g
    assert (stmt, ONTO["role"], None) in g


def test_reification_bnode_is_deterministic_across_runs() -> None:
    g1 = dataset_to_rdf(_dataset())
    g2 = dataset_to_rdf(_dataset())
    assert set(g1) == set(g2)
    assert encode_dataset(_dataset(), "ttl") == encode_dataset(_dataset(), "ttl")
    assert encode_dataset(_dataset(), "nt") == encode_dataset(_dataset(), "nt")


def test_turtle_round_trips_via_rdflib() -> None:
    data = encode_dataset(_dataset(), "ttl")
    parsed = Graph().parse(data=data.decode("utf-8"), format="turtle")
    assert len(parsed) == len(dataset_to_rdf(_dataset()))


def test_ntriples_round_trips_via_rdflib() -> None:
    data = encode_dataset(_dataset(), "nt")
    parsed = Graph().parse(data=data.decode("utf-8"), format="nt")
    assert len(parsed) == len(dataset_to_rdf(_dataset()))


def test_jsonld_round_trips_via_rdflib() -> None:
    data = encode_dataset(_dataset(), "jsonld")
    parsed = Graph().parse(data=data.decode("utf-8"), format="json-ld")
    assert len(parsed) == len(dataset_to_rdf(_dataset()))


def test_rdfxml_round_trips_via_rdflib() -> None:
    data = encode_dataset(_dataset(), "rdfxml")
    parsed = Graph().parse(data=data.decode("utf-8"), format="xml")
    assert len(parsed) == len(dataset_to_rdf(_dataset()))
