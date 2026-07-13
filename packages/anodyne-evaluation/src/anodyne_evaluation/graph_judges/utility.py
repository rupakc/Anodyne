"""Graph-utility expert: a lightweight GNN-style TSTR proxy (no torch-geometric).

Trains a small, seeded sklearn classifier to predict a node's **ontology type**
from cheap structural features + numeric node properties, then measures
Train-on-Synthetic / Test-on-Real transfer:

  features per node = [degree, clustering, triangle count, avg-neighbour degree,
                       community size] + shared numeric node properties
  label            = node.type

The efficacy ratio TSTR / TRTR (both scored on the real graph) approximates how
well structure+attribute relationships learned on the synthetic graph transfer
to reality. Requires a reference graph and >= 2 node types (else not applicable).
Deterministic: seeded `RandomForestClassifier`, feature order fixed by sorting.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx  # type: ignore[import-untyped]
import numpy as np
from sklearn.ensemble import RandomForestClassifier  # type: ignore[import-untyped]
from sklearn.metrics import accuracy_score  # type: ignore[import-untyped]

from anodyne_evaluation.graph_judges.base import GraphJudge, clamp01, require_graph, to_undirected
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext, JudgeNotApplicable

if TYPE_CHECKING:
    from anodyne_graph.models import GraphDataset

_STRUCTURAL = ("degree", "clustering", "triangles", "avg_neighbor_degree", "community_size")


def _numeric_property_keys(dataset: GraphDataset) -> list[str]:
    keys: set[str] = set()
    for node in dataset.nodes:
        for key, value in node.properties.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                keys.add(key)
    return sorted(keys)


def _community_size(g: nx.Graph) -> dict[str, int]:
    sizes: dict[str, int] = {}
    if g.number_of_edges() == 0:
        return {node: 1 for node in g.nodes()}
    try:
        communities = nx.community.greedy_modularity_communities(g)
    except (ValueError, ZeroDivisionError, KeyError):
        return {node: 1 for node in g.nodes()}
    for community in communities:
        for node in community:
            sizes[node] = len(community)
    for node in g.nodes():
        sizes.setdefault(node, 1)
    return sizes


def _features_labels(dataset: GraphDataset, prop_keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    g = to_undirected(dataset)
    clustering = nx.clustering(g)
    triangles = nx.triangles(g)
    avg_neighbor = nx.average_neighbor_degree(g)
    community_size = _community_size(g)
    rows: list[list[float]] = []
    labels: list[str] = []
    for node in dataset.nodes:
        if not g.has_node(node.id):
            continue
        struct = [
            float(g.degree(node.id)),
            float(clustering.get(node.id, 0.0)),
            float(triangles.get(node.id, 0)),
            float(avg_neighbor.get(node.id, 0.0)),
            float(community_size.get(node.id, 1)),
        ]
        props = [
            float(node.properties.get(k))  # type: ignore[arg-type]
            if isinstance(node.properties.get(k), (int, float))
            and not isinstance(node.properties.get(k), bool)
            else 0.0
            for k in prop_keys
        ]
        rows.append(struct + props)
        labels.append(node.type)
    return np.asarray(rows, dtype=float), np.asarray(labels, dtype=object)


def _fit_score(
    xtr: np.ndarray, ytr: np.ndarray, xte: np.ndarray, yte: np.ndarray, seed: int
) -> float:
    if len(np.unique(ytr)) < 2:
        pred = np.full(shape=len(yte), fill_value=ytr[0])
        return float(accuracy_score(yte.astype(str), pred.astype(str)))
    model = RandomForestClassifier(n_estimators=25, random_state=seed)
    model.fit(xtr, ytr.astype(str))
    return float(accuracy_score(yte.astype(str), model.predict(xte)))


class GraphUtilityJudge(GraphJudge):
    dimension = EvalDimension.GRAPH_UTILITY

    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        syn = require_graph(ctx)
        if ctx.reference_graph is None:
            raise JudgeNotApplicable("graph utility (TSTR) requires a reference graph")
        ref = ctx.reference_graph

        prop_keys = sorted(set(_numeric_property_keys(syn)) & set(_numeric_property_keys(ref)))
        xs, ys = _features_labels(syn, prop_keys)
        xr, yr = _features_labels(ref, prop_keys)
        if len(xs) < 2 or len(xr) < 2:
            raise JudgeNotApplicable("graph utility (TSTR) requires >= 2 nodes per graph")
        if len(np.unique(yr)) < 2 or len(np.unique(ys)) < 2:
            raise JudgeNotApplicable("graph utility (TSTR) requires >= 2 node types to classify")

        tstr = _fit_score(xs, ys, xr, yr, ctx.seed)
        trtr = _fit_score(xr, yr, xr, yr, ctx.seed)
        ratio = clamp01(tstr / trtr) if trtr > 0 else 0.0
        recs: list[str] = []
        if ratio < 0.7:
            recs.append(
                "Classifiers trained on the synthetic graph transfer poorly to real; "
                "structure/attribute relationships differ."
            )
        return ExpertScore(
            dimension=self.dimension,
            score=ratio,
            rationale=(
                f"Node-type TSTR over {len(_STRUCTURAL)} structural + {len(prop_keys)} "
                f"property feature(s): TSTR={tstr:.3f}, TRTR={trtr:.3f}, ratio={ratio:.3f}."
            ),
            metrics={
                "tstr_accuracy": tstr,
                "trtr_accuracy": trtr,
                "efficacy_ratio": ratio,
                "feature_count": float(len(_STRUCTURAL) + len(prop_keys)),
            },
            recommendations=recs,
        )
