"""Privacy / leakage expert: structural re-identification risk vs a reference.

Heuristic (documented, not a formal privacy guarantee — DP for graphs is an
explicit future track per the spec). Two structural leakage signals:

- **node re-identification** — each node gets a structural *fingerprint*
  ``(degree, sorted tuple of neighbour degrees, clustering rounded to 2dp)``.
  The re-identification rate is the fraction of synthetic nodes whose
  fingerprint exactly matches some reference node's fingerprint — a proxy for
  "this synthetic node is structurally a copy of a real one".
- **verbatim-edge overlap** — the fraction of synthetic edges whose endpoint
  degree-signature ``frozenset({deg(u), deg(v)})`` matches a reference edge,
  approximating copied local structure.

``leakage = mean(reident_rate, edge_overlap)``; ``score = 1 - leakage`` (higher
== safer). Two *identical* graphs therefore score near 0 (maximal leakage); a
statistically-similar but genuinely resampled graph scores high. Requires a
reference graph.
"""

from __future__ import annotations

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from anodyne_evaluation.graph_judges.base import GraphJudge, clamp01, require_graph, to_undirected
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable


def _fingerprints(g: nx.Graph) -> set[tuple[int, tuple[int, ...], float]]:
    clustering = nx.clustering(g)
    out: set[tuple[int, tuple[int, ...], float]] = set()
    for node in g.nodes():
        neighbor_degrees = tuple(sorted(g.degree(nbr) for nbr in g.neighbors(node)))
        out.add((g.degree(node), neighbor_degrees, round(float(clustering.get(node, 0.0)), 2)))
    return out


def _node_fingerprint(g: nx.Graph, node: str) -> tuple[int, tuple[int, ...], float]:
    clustering = nx.clustering(g, node)
    neighbor_degrees = tuple(sorted(g.degree(nbr) for nbr in g.neighbors(node)))
    return (g.degree(node), neighbor_degrees, round(float(clustering), 2))


def _edge_signatures(g: nx.Graph) -> set[frozenset[int]]:
    return {frozenset((g.degree(u), g.degree(v))) for u, v in g.edges()}


class GraphPrivacyJudge(GraphJudge):
    dimension = EvalDimension.GRAPH_PRIVACY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        syn = require_graph(ctx)
        if ctx.reference_graph is None:
            raise JudgeNotApplicable("graph privacy/leakage requires a reference graph")
        gs = to_undirected(syn)
        gr = to_undirected(ctx.reference_graph)
        if gs.number_of_nodes() == 0 or gr.number_of_nodes() == 0:
            raise JudgeNotApplicable("graph privacy/leakage requires non-empty graphs")

        ref_fingerprints = _fingerprints(gr)
        matched = sum(1 for node in gs.nodes() if _node_fingerprint(gs, node) in ref_fingerprints)
        reident_rate = matched / gs.number_of_nodes()

        if gs.number_of_edges() and gr.number_of_edges():
            ref_edge_sigs = _edge_signatures(gr)
            edge_matched = sum(
                1 for u, v in gs.edges() if frozenset((gs.degree(u), gs.degree(v))) in ref_edge_sigs
            )
            edge_overlap = edge_matched / gs.number_of_edges()
        else:
            edge_overlap = 0.0

        leakage = float(np.mean([reident_rate, edge_overlap]))
        score = clamp01(1.0 - leakage)
        recs: list[str] = []
        if reident_rate > 0.5:
            recs.append(
                "Many synthetic nodes are structurally identical to reference nodes; "
                "increase topology resampling to reduce re-identification risk."
            )
        if edge_overlap > 0.5:
            recs.append("High verbatim local-structure overlap with the reference graph.")
        return ExpertScore(
            dimension=self.dimension,
            score=score,
            rationale=(
                f"Structural leakage: node re-identification {reident_rate:.3f}, "
                f"verbatim-edge overlap {edge_overlap:.3f} -> privacy {score:.3f}."
            ),
            metrics={
                "reidentification_rate": reident_rate,
                "edge_overlap": edge_overlap,
                "leakage": leakage,
            },
            recommendations=recs,
        )
