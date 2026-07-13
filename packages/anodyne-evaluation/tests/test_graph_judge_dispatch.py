from __future__ import annotations

from typing import Any
from uuid import uuid4

import networkx as nx  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import Modality
from anodyne_evaluation.evaluator import MoEEvaluator, graph_judges, judges_for_modality
from anodyne_evaluation.graph_judges import (
    GraphJudge,
    SemanticPlausibilityGraphJudge,
)
from anodyne_evaluation.judges import FidelityJudge
from anodyne_evaluation.models import GRAPH_WEIGHTS, EvalDimension


def test_dispatch_selects_graph_judges_for_graph_modality() -> None:
    graph = judges_for_modality(Modality.GRAPH)
    assert graph and all(isinstance(j, GraphJudge) for j in graph)
    assert {j.dimension for j in graph} <= set(GRAPH_WEIGHTS)
    # No LLM provider => the semantic (LLM) expert is omitted.
    assert not any(isinstance(j, SemanticPlausibilityGraphJudge) for j in graph)


def test_dispatch_keeps_tabular_judges_for_other_modalities() -> None:
    tabular = judges_for_modality(Modality.TABULAR)
    assert any(isinstance(j, FidelityJudge) for j in tabular)
    assert not any(isinstance(j, GraphJudge) for j in tabular)


async def test_graph_judges_aggregate_into_360_report(
    dataset_from_nx: Any, graph_context: Any
) -> None:
    g = nx.barabasi_albert_graph(40, 2, seed=2)

    def by_degree(_n: object, d: int) -> str:
        return "Hub" if d >= 3 else "Leaf"

    subject = dataset_from_nx(g, node_type_fn=by_degree)
    reference = dataset_from_nx(g.copy(), node_type_fn=by_degree)
    ctx = graph_context(subject, reference)

    evaluator = MoEEvaluator(graph_judges())  # statistical graph experts only
    report = await evaluator.evaluate(
        ctx,
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        reference_version_id=uuid4(),
    )
    dims = {s.dimension for s in report.expert_scores}
    assert EvalDimension.GRAPH_STRUCTURE in dims
    assert EvalDimension.GRAPH_ONTOLOGY in dims
    assert 0.0 <= report.overall_score <= 1.0
    # Weights renormalize over exactly the dimensions that produced a score.
    assert sum(report.weights.values()) == pytest.approx(1.0)
    assert set(report.weights) == {str(d) for d in dims}
    # An identical-to-reference graph is structurally strong.
    structure = next(
        s for s in report.expert_scores if s.dimension is EvalDimension.GRAPH_STRUCTURE
    )
    assert structure.score > 0.9
