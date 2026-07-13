"""Connectivity / coverage expert: intrinsic structural health (no reference).

Four intrinsic signals averaged into a 0..1 score (higher == healthier):
- **giant-component fraction** — nodes in the largest connected component / total
  (a well-formed graph is mostly one component);
- **connectedness** — ``1 - isolated-node fraction`` (degree-0 nodes are usually
  a defect);
- **relation-type coverage** — fraction of the ontology's edge types that the
  instance data actually uses;
- **node-type coverage** — fraction of the ontology's node types instantiated.

Intrinsic, so it needs no reference graph — it always applies when a graph is
present.
"""

from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from anodyne_evaluation.graph_judges.base import GraphJudge, clamp01, require_graph, to_undirected
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


class ConnectivityCoverageGraphJudge(GraphJudge):
    dimension = EvalDimension.GRAPH_CONNECTIVITY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        graph = require_graph(ctx)
        g = to_undirected(graph)
        n = g.number_of_nodes()
        if n == 0:
            raise JudgeNotApplicable("connectivity requires a non-empty graph")

        giant = max((len(c) for c in nx.connected_components(g)), default=0)
        giant_fraction = giant / n
        isolated = sum(1 for _, d in g.degree() if d == 0)
        isolated_fraction = isolated / n

        onto_edge_types = {et.name for et in graph.ontology.edge_types}
        present_edge_types = {e.type for e in graph.edges}
        relation_coverage = (
            len(present_edge_types & onto_edge_types) / len(onto_edge_types)
            if onto_edge_types
            else 1.0
        )
        onto_node_types = {nt.name for nt in graph.ontology.node_types}
        present_node_types = {node.type for node in graph.nodes}
        node_coverage = (
            len(present_node_types & onto_node_types) / len(onto_node_types)
            if onto_node_types
            else 1.0
        )

        score = clamp01(
            float(
                np.mean([giant_fraction, 1.0 - isolated_fraction, relation_coverage, node_coverage])
            )
        )
        recs: list[str] = []
        if isolated_fraction > 0.05:
            recs.append(f"{isolated} isolated node(s); connect or prune them.")
        if giant_fraction < 0.8:
            recs.append("Graph is fragmented into many components; increase inter-cluster edges.")
        if relation_coverage < 1.0:
            recs.append("Some ontology relation types are never instantiated.")
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=(
                f"Giant component {giant_fraction:.3f}, isolated {isolated_fraction:.3f}, "
                f"relation-type coverage {relation_coverage:.3f}, node-type coverage "
                f"{node_coverage:.3f}."
            ),
            metrics={
                "giant_component_fraction": giant_fraction,
                "isolated_node_fraction": isolated_fraction,
                "relation_type_coverage": relation_coverage,
                "node_type_coverage": node_coverage,
                "node_count": float(n),
            },
            recommendations=recs,
        )
