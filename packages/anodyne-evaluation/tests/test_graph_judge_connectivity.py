from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.graph_judges.connectivity import ConnectivityCoverageGraphJudge
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import JudgeNotApplicable


async def test_connected_fully_covered_graph_scores_high(
    dataset_from_nx: Any, graph_context: Any
) -> None:
    graph = dataset_from_nx(nx.cycle_graph(10))  # one component, no isolated nodes
    score = await ConnectivityCoverageGraphJudge().evaluate(graph_context(graph))
    assert score.dimension is EvalDimension.GRAPH_CONNECTIVITY
    assert score.metrics["giant_component_fraction"] == 1.0
    assert score.metrics["isolated_node_fraction"] == 0.0
    assert score.metrics["relation_type_coverage"] == 1.0
    assert score.score == pytest.approx(1.0)


async def test_isolated_nodes_and_missing_relation_lower_score(
    dataset_from_nx: Any, graph_context: Any
) -> None:
    g = nx.cycle_graph(8)
    g.add_nodes_from([100, 101])  # two isolated nodes
    # ontology declares an extra relation type that is never instantiated
    graph = dataset_from_nx(g, extra_edge_types=["UNUSED"])
    score = await ConnectivityCoverageGraphJudge().evaluate(graph_context(graph))
    assert score.metrics["isolated_node_fraction"] > 0.0
    assert score.metrics["relation_type_coverage"] < 1.0
    assert score.score < 1.0
    assert score.recommendations


async def test_empty_graph_not_applicable(dataset_from_nx: Any, graph_context: Any) -> None:
    graph = dataset_from_nx(nx.empty_graph(0))
    with pytest.raises(JudgeNotApplicable):
        await ConnectivityCoverageGraphJudge().evaluate(graph_context(graph))
