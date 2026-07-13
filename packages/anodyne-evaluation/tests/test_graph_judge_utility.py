from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.graph_judges.utility import GraphUtilityJudge
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import JudgeNotApplicable


def _by_degree(_node: object, degree: int) -> str:
    return "Hub" if degree >= 3 else "Leaf"


async def test_tstr_transfers_on_identical_graphs(dataset_from_nx: Any, graph_context: Any) -> None:
    g = nx.barabasi_albert_graph(60, 2, seed=3)
    subject = dataset_from_nx(g, node_type_fn=_by_degree)
    reference = dataset_from_nx(g.copy(), node_type_fn=_by_degree)
    score = await GraphUtilityJudge().evaluate(graph_context(subject, reference, seed=0))
    assert score.dimension is EvalDimension.GRAPH_UTILITY
    # Train-on-synthetic (== real copy) transfers to real: efficacy near parity.
    assert score.score > 0.6
    assert score.metrics["tstr_accuracy"] > 0.0


async def test_requires_reference(dataset_from_nx: Any, graph_context: Any) -> None:
    subject = dataset_from_nx(nx.barabasi_albert_graph(20, 2, seed=1), node_type_fn=_by_degree)
    with pytest.raises(JudgeNotApplicable):
        await GraphUtilityJudge().evaluate(graph_context(subject))


async def test_single_node_type_not_applicable(dataset_from_nx: Any, graph_context: Any) -> None:
    g = nx.barabasi_albert_graph(20, 2, seed=1)
    subject = dataset_from_nx(g)  # every node is "Item"
    reference = dataset_from_nx(g.copy())
    with pytest.raises(JudgeNotApplicable):
        await GraphUtilityJudge().evaluate(graph_context(subject, reference))
