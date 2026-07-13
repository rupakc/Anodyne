"""Ontology-consistency expert: constraint validation pass-rate over the graph.

**Why a pure-Python checker, not pySHACL:** the canonical model is a *labelled
property graph*, not RDF. Running SHACL would first require projecting the LPG to
RDF (reifying edge properties), pulling in `rdflib` + `pyshacl` — heavyweight and
lossy — only to re-express the very domain/range/datatype rules the LPG ontology
already states natively. So we validate the graph *directly* against its
`GraphOntology`: every check is one boolean, the score is the pass fraction.

Checks performed (each contributes one unit to the denominator):
- node type is declared in the ontology;
- each non-nullable node property is present, and every present property
  matches its declared datatype + `choices`/`min`/`max` constraints;
- edge type is declared, its endpoints resolve to existing nodes, and their
  types satisfy the relation's declared domain/range (`source_type`/`target_type`).

*Cardinality:* the frozen GA ontology model carries no relation-cardinality
fields, so cardinality is a documented no-op here (property-level `min`/`max` on
values are checked); it activates for free once the ontology model grows them.
"""

from __future__ import annotations

from typing import Any

from anodyne_evaluation.graph_judges.base import GraphJudge, require_graph
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext


def _datatype_ok(value: Any, datatype: str) -> bool:
    dt = datatype.lower()
    if dt == "boolean":
        return isinstance(value, bool)
    if dt == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if dt == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if dt in ("string", "datetime"):
        return isinstance(value, str)
    return True  # unknown/extended datatype: not our place to reject


def _constraints_ok(value: Any, constraints: dict[str, Any]) -> bool:
    choices = constraints.get("choices")
    if isinstance(choices, list) and value not in choices:
        return False
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        lo, hi = constraints.get("min"), constraints.get("max")
        if isinstance(lo, (int, float)) and value < lo:
            return False
        if isinstance(hi, (int, float)) and value > hi:
            return False
    return True


class OntologyConsistencyGraphJudge(GraphJudge):
    dimension = EvalDimension.GRAPH_ONTOLOGY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        graph = require_graph(ctx)
        onto = graph.ontology
        by_id = {n.id: n for n in graph.nodes}

        checks = 0
        passed = 0
        node_type_violations = 0
        property_violations = 0
        edge_violations = 0

        for node in graph.nodes:
            checks += 1
            node_type = onto.node_type(node.type)
            if node_type is None:
                node_type_violations += 1
                continue
            passed += 1
            for prop in node_type.properties:
                checks += 1
                value = node.properties.get(prop.name)
                if value is None:
                    if prop.nullable:
                        passed += 1
                    else:
                        property_violations += 1
                    continue
                if _datatype_ok(value, prop.datatype) and _constraints_ok(value, prop.constraints):
                    passed += 1
                else:
                    property_violations += 1

        for edge in graph.edges:
            checks += 1
            edge_type = onto.edge_type(edge.type)
            src, tgt = by_id.get(edge.source), by_id.get(edge.target)
            if (
                edge_type is not None
                and src is not None
                and tgt is not None
                and src.type == edge_type.source_type
                and tgt.type == edge_type.target_type
            ):
                passed += 1
            else:
                edge_violations += 1

        pass_fraction = passed / checks if checks else 1.0
        recs: list[str] = []
        if node_type_violations:
            recs.append(f"{node_type_violations} node(s) use a type absent from the ontology.")
        if property_violations:
            recs.append(
                f"{property_violations} node propert(ies) violate datatype/required/choice rules."
            )
        if edge_violations:
            recs.append(
                f"{edge_violations} edge(s) violate the relation's domain/range or reference "
                "a missing endpoint."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=pass_fraction,
            rationale=(
                f"Ontology constraint pass-rate {pass_fraction:.3f} over {checks} checks "
                f"({node_type_violations} node-type, {property_violations} property, "
                f"{edge_violations} edge violations)."
            ),
            metrics={
                "pass_fraction": pass_fraction,
                "total_checks": float(checks),
                "node_type_violations": float(node_type_violations),
                "property_violations": float(property_violations),
                "edge_violations": float(edge_violations),
            },
            recommendations=recs,
        )
