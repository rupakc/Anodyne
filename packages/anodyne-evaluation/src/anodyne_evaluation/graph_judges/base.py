"""Shared helpers for the graph expert judges (sub-system GD).

`GraphJudge` mirrors `judges.base.StatisticalJudge`: the CPU-bound math lives in
a synchronous `compute`, and the async `evaluate` just calls it so the judge
satisfies the same `Judge` port as every other expert (and so the Ray runner can
dispatch `compute` as a remote task). The LLM-backed semantic judge overrides
`evaluate` directly instead.

Graphs are converted to a **simple undirected** ``networkx.Graph`` for the
statistical/spectral metrics (clustering, assortativity, modularity, the
Laplacian spectrum) — parallel edges collapse and self-loops are dropped, which
is what those metrics assume. Edges are only added when both endpoints exist as
declared nodes, so a dataset with dangling references degrades gracefully rather
than raising.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

import networkx as nx  # type: ignore[import-untyped]
import numpy as np

from anodyne_evaluation.models import ExpertScore
from anodyne_evaluation.ports import EvaluationContext, Judge, JudgeNotApplicable

if TYPE_CHECKING:
    from anodyne_graph.models import GraphDataset, Node


def clamp01(x: float) -> float:
    """Clamp to [0, 1]; NaN/inf collapse to 0.0 so a degenerate metric never
    yields an out-of-range or non-finite score."""
    if not np.isfinite(x):
        return 0.0
    return float(min(1.0, max(0.0, x)))


def require_graph(ctx: EvaluationContext) -> GraphDataset:
    """Return the subject graph, or raise `JudgeNotApplicable` if this run has none."""
    if ctx.subject_graph is None:
        raise JudgeNotApplicable("graph judge requires a graph artifact under evaluation")
    return ctx.subject_graph


def to_undirected(dataset: GraphDataset) -> nx.Graph:
    """Build a simple undirected `networkx.Graph` from an LPG dataset.

    Node ``type`` is stored as the ``ntype`` attribute; edge ``type`` as
    ``etype``. Self-loops and duplicate edges are collapsed (the statistical
    metrics operate on simple graphs).
    """
    g: nx.Graph = nx.Graph()
    for node in dataset.nodes:
        g.add_node(node.id, ntype=node.type)
    for edge in dataset.edges:
        if edge.source == edge.target:
            continue
        if g.has_node(edge.source) and g.has_node(edge.target):
            g.add_edge(edge.source, edge.target, etype=edge.type)
    return g


def node_label(node: Node) -> str:
    """A human-readable label for a node: a ``name``/``label``/``title`` property
    if present, else the first string-valued property, else the node id."""
    props = node.properties
    for key in ("name", "label", "title"):
        val = props.get(key)
        if isinstance(val, str) and val:
            return val
    for val in props.values():
        if isinstance(val, str) and val:
            return val
    return node.id


class GraphJudge(Judge):
    @abstractmethod
    def compute(self, ctx: EvaluationContext) -> ExpertScore:
        """Synchronous scoring; may raise `JudgeNotApplicable`."""

    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
        return self.compute(ctx)
