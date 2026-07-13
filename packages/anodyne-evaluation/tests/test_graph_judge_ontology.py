from __future__ import annotations

from typing import Any

from anodyne_evaluation.graph_judges.ontology import OntologyConsistencyGraphJudge
from anodyne_evaluation.models import EvalDimension
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
    PropertySpec,
)


def _ontology() -> GraphOntology:
    return GraphOntology(
        node_types=[
            NodeType(
                name="Person",
                properties=[
                    PropertySpec(name="age", datatype="integer"),
                    PropertySpec(
                        name="role", datatype="string", constraints={"choices": ["dev", "pm"]}
                    ),
                ],
            ),
            NodeType(name="Company"),
        ],
        edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
    )


async def test_valid_graph_passes_fully(graph_context: Any) -> None:
    onto = _ontology()
    graph = GraphDataset(
        ontology=onto,
        nodes=[
            Node(id="p1", type="Person", properties={"age": 30, "role": "dev"}),
            Node(id="c1", type="Company"),
        ],
        edges=[Edge(id="e1", type="WORKS_AT", source="p1", target="c1")],
    )
    score = await OntologyConsistencyGraphJudge().evaluate(graph_context(graph))
    assert score.dimension is EvalDimension.GRAPH_ONTOLOGY
    assert score.score == 1.0
    assert score.metrics["pass_fraction"] == 1.0


async def test_violations_lower_pass_rate(graph_context: Any) -> None:
    onto = _ontology()
    graph = GraphDataset(
        ontology=onto,
        nodes=[
            # bad datatype (age is a string), bad choice (role), unknown node type
            Node(id="p1", type="Person", properties={"age": "old", "role": "ceo"}),
            Node(id="c1", type="Company"),
            Node(id="x1", type="Alien"),
        ],
        edges=[
            # domain/range violation: WORKS_AT should be Person->Company
            Edge(id="e1", type="WORKS_AT", source="c1", target="p1"),
            # unknown edge type
            Edge(id="e2", type="OWNS", source="p1", target="c1"),
        ],
    )
    score = await OntologyConsistencyGraphJudge().evaluate(graph_context(graph))
    assert score.score < 1.0
    assert score.metrics["node_type_violations"] == 1.0
    assert score.metrics["property_violations"] == 2.0
    assert score.metrics["edge_violations"] == 2.0
    assert score.recommendations
