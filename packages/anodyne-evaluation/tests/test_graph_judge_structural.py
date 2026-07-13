from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.graph_judges.structural import StructuralFidelityGraphJudge
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import JudgeNotApplicable


async def test_identical_graphs_score_near_one(dataset_from_nx: Any, graph_context: Any) -> None:
    g = nx.barabasi_albert_graph(40, 2, seed=7)
    subject = dataset_from_nx(g)
    reference = dataset_from_nx(g.copy())
    score = await StructuralFidelityGraphJudge().evaluate(graph_context(subject, reference))
    assert score.dimension is EvalDimension.GRAPH_STRUCTURE
    assert score.score > 0.95
    assert score.metrics["degree_ks"] == pytest.approx(0.0, abs=1e-9)
    assert score.metrics["spectral_distance"] == pytest.approx(0.0, abs=1e-9)


async def test_injected_edges_lower_fidelity(dataset_from_nx: Any, graph_context: Any) -> None:
    base = nx.barabasi_albert_graph(40, 2, seed=7)
    perturbed = base.copy()
    rng = nx.utils.create_random_state(11)
    nodes = list(perturbed.nodes())
    added = 0
    while added < 60:
        u, v = rng.choice(nodes, 2, replace=False)
        if not perturbed.has_edge(u, v):
            perturbed.add_edge(u, v)
            added += 1

    identical = await StructuralFidelityGraphJudge().evaluate(
        graph_context(dataset_from_nx(base), dataset_from_nx(base.copy()))
    )
    noisy = await StructuralFidelityGraphJudge().evaluate(
        graph_context(dataset_from_nx(perturbed), dataset_from_nx(base))
    )
    assert noisy.score < identical.score
    assert noisy.metrics["degree_ks"] > 0.0


async def test_requires_reference(dataset_from_nx: Any, graph_context: Any) -> None:
    subject = dataset_from_nx(nx.path_graph(5))
    with pytest.raises(JudgeNotApplicable):
        await StructuralFidelityGraphJudge().evaluate(graph_context(subject))
