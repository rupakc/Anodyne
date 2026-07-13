from __future__ import annotations

from typing import Any

import networkx as nx  # type: ignore[import-untyped]
import pytest
from anodyne_evaluation.graph_judges.privacy import GraphPrivacyJudge
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import JudgeNotApplicable


async def test_identical_graph_leaks(dataset_from_nx: Any, graph_context: Any) -> None:
    g = nx.barabasi_albert_graph(40, 2, seed=5)
    subject = dataset_from_nx(g)
    reference = dataset_from_nx(g.copy())
    score = await GraphPrivacyJudge().evaluate(graph_context(subject, reference))
    assert score.dimension is EvalDimension.GRAPH_PRIVACY
    # A verbatim copy is maximally leaky -> low privacy score, high re-identification.
    assert score.metrics["reidentification_rate"] > 0.5
    assert score.score < 0.5


async def test_resampled_graph_is_safer_than_copy(dataset_from_nx: Any, graph_context: Any) -> None:
    reference = nx.barabasi_albert_graph(60, 3, seed=5)
    resampled = nx.barabasi_albert_graph(60, 3, seed=99)  # same model, different draw
    copy_score = await GraphPrivacyJudge().evaluate(
        graph_context(dataset_from_nx(reference.copy()), dataset_from_nx(reference))
    )
    resampled_score = await GraphPrivacyJudge().evaluate(
        graph_context(dataset_from_nx(resampled), dataset_from_nx(reference))
    )
    assert resampled_score.score > copy_score.score


async def test_requires_reference(dataset_from_nx: Any, graph_context: Any) -> None:
    subject = dataset_from_nx(nx.path_graph(5))
    with pytest.raises(JudgeNotApplicable):
        await GraphPrivacyJudge().evaluate(graph_context(subject))
