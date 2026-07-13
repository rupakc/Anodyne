"""Wave-2 follow-up fixes for graph export (see docs/graph-followups.md #3, #4)."""

from __future__ import annotations

from anodyne_graph.export import ONTO, dataset_to_cypher, dataset_to_rdf
from anodyne_graph.models import (
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
)
from rdflib import Literal
from rdflib.namespace import XSD


def _dt_dataset() -> GraphDataset:
    ontology = GraphOntology(
        node_types=[
            NodeType(
                name="Event",
                properties=[
                    PropertySpec(name="occurred_at", datatype="datetime"),
                    PropertySpec(name="label", datatype="string"),
                ],
            )
        ],
        edge_types=[EdgeType(name="FOLLOWS", source_type="Event", target_type="Event")],
    )
    nodes = [
        Node(
            id="ev1",
            type="Event",
            properties={"occurred_at": "2026-07-13T10:00:00", "label": "line1\nline2\tend"},
        )
    ]
    return GraphDataset(ontology=ontology, nodes=nodes, edges=[])


def test_cypher_escapes_control_characters() -> None:
    script = dataset_to_cypher(_dt_dataset()).decode("utf-8")
    # No raw control characters survive into the emitted script.
    assert "\n" in script  # statement separators are fine
    # ... but not inside a value: the label's raw newline/tab must be escaped.
    assert "line1\\nline2\\tend" in script
    assert "line1\nline2\tend" not in script


def test_rdf_datetime_literal_typed_as_xsd_datetime() -> None:
    g = dataset_to_rdf(_dt_dataset())
    values = list(g.objects(predicate=ONTO["occurred_at"]))
    assert values == [Literal("2026-07-13T10:00:00", datatype=XSD.dateTime)]
