from __future__ import annotations

from anodyne_graph.export import ONTO, encode_dataset, ontology_to_owl
from anodyne_graph.models import EdgeType, GraphOntology, NodeType, PropertySpec
from rdflib import Graph
from rdflib.namespace import OWL, RDF, RDFS


def _ontology() -> GraphOntology:
    return GraphOntology(
        node_types=[
            NodeType(
                name="Person",
                properties=[
                    PropertySpec(name="age", datatype="integer"),
                    PropertySpec(name="name", datatype="string"),
                ],
            ),
            NodeType(name="Company"),
        ],
        edge_types=[
            EdgeType(
                name="WORKS_AT",
                source_type="Person",
                target_type="Company",
                properties=[PropertySpec(name="since", datatype="integer")],
            )
        ],
        subclass_of={"Person": "Agent"},
    )


def test_node_types_become_owl_classes() -> None:
    g = ontology_to_owl(_ontology())
    assert (ONTO["Person"], RDF.type, OWL.Class) in g
    assert (ONTO["Company"], RDF.type, OWL.Class) in g


def test_subclass_of_is_preserved() -> None:
    g = ontology_to_owl(_ontology())
    assert (ONTO["Person"], RDFS.subClassOf, ONTO["Agent"]) in g


def test_edge_types_become_object_properties_with_domain_range() -> None:
    g = ontology_to_owl(_ontology())
    assert (ONTO["WORKS_AT"], RDF.type, OWL.ObjectProperty) in g
    assert (ONTO["WORKS_AT"], RDFS.domain, ONTO["Person"]) in g
    assert (ONTO["WORKS_AT"], RDFS.range, ONTO["Company"]) in g


def test_property_specs_become_datatype_properties() -> None:
    g = ontology_to_owl(_ontology())
    assert (ONTO["Person.age"], RDF.type, OWL.DatatypeProperty) in g
    assert (ONTO["Person.age"], RDFS.domain, ONTO["Person"]) in g
    assert (ONTO["WORKS_AT.since"], RDF.type, OWL.DatatypeProperty) in g


def test_owl_serialization_round_trips_via_rdflib() -> None:
    data = encode_dataset_for_ontology()
    parsed = Graph().parse(data=data.decode("utf-8"), format="xml")
    assert len(parsed) == len(ontology_to_owl(_ontology()))
    assert (ONTO["Person"], RDF.type, OWL.Class) in parsed


def encode_dataset_for_ontology() -> bytes:
    # `encode_dataset` takes a `GraphDataset`; build a minimal one carrying the
    # ontology under test, since `owl` export only reads `dataset.ontology`.
    from anodyne_graph.models import GraphDataset, compute_metrics

    ds = GraphDataset(ontology=_ontology(), nodes=[], edges=[], metrics=compute_metrics([], []))
    return encode_dataset(ds, "owl")
