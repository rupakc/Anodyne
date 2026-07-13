"""Wave-2 follow-up fixes for the graph judges (see docs/graph-followups.md #1, #2)."""

from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.graph_judges.ontology import OntologyConsistencyGraphJudge
from anodyne_evaluation.graph_judges.structural import _spectral_distance
from anodyne_evaluation.ports import JudgeNotApplicable
from anodyne_graph.models import GraphDataset, GraphOntology


def test_spectral_distance_identical_is_zero() -> None:
    g = nx.barabasi_albert_graph(60, 2, seed=3)
    assert _spectral_distance(g, g.copy()) == pytest.approx(0.0, abs=1e-9)


def test_spectral_distance_not_inflated_by_size_gap() -> None:
    # Same generative structure, very different node counts. The old zero-padding
    # let the size gap dominate (distance ~0.45); resampling to a common length
    # keeps a structurally-similar pair close.
    small = nx.barabasi_albert_graph(50, 2, seed=3)
    large = nx.barabasi_albert_graph(250, 2, seed=3)
    assert _spectral_distance(small, large) < 0.3


async def test_empty_graph_ontology_is_not_applicable(graph_context: Any) -> None:
    empty = GraphDataset(ontology=GraphOntology(node_types=[], edge_types=[]), nodes=[], edges=[])
    with pytest.raises(JudgeNotApplicable):
        await OntologyConsistencyGraphJudge().evaluate(graph_context(empty))
