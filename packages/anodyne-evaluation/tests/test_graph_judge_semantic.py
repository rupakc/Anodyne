from __future__ import annotations

from typing import Any

import pytest
from anodyne_core.models import LLMRequest, LLMResponse, Usage
from anodyne_evaluation.graph_judges.semantic import (
    SemanticPlausibilityError,
    SemanticPlausibilityGraphJudge,
)
from anodyne_evaluation.models import EvalDimension
from anodyne_evaluation.ports import JudgeNotApplicable
from anodyne_graph.models import (
    Edge,
    EdgeType,
    GraphDataset,
    GraphOntology,
    Node,
    NodeType,
)


class _FakeProvider:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(content=self._content, usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _graph() -> GraphDataset:
    onto = GraphOntology(
        node_types=[NodeType(name="Person"), NodeType(name="Company")],
        edge_types=[EdgeType(name="WORKS_AT", source_type="Person", target_type="Company")],
    )
    return GraphDataset(
        ontology=onto,
        nodes=[
            Node(id="p1", type="Person", properties={"name": "Ada Lovelace"}),
            Node(id="c1", type="Company", properties={"name": "Analytical Engines Ltd"}),
        ],
        edges=[Edge(id="e1", type="WORKS_AT", source="p1", target="c1")],
    )


async def test_parses_rubric_and_renders_triples(graph_context: Any, model_cfg: Any) -> None:
    provider = _FakeProvider(
        '```json\n{"realism": 5, "coherence": 4, "rationale": "plausible"}\n```'
    )
    judge = SemanticPlausibilityGraphJudge(provider, model_cfg)  # type: ignore[arg-type]
    score = await judge.evaluate(graph_context(_graph()))
    assert score.dimension is EvalDimension.GRAPH_SEMANTIC
    assert score.score == pytest.approx((5 + 4) / 10.0)
    assert score.metrics == {"realism": 5.0, "coherence": 4.0}
    # The rendered triples reached the prompt with entity labels + relation.
    assert provider.last_request is not None
    prompt = provider.last_request.messages[-1].content
    assert "Ada Lovelace" in prompt and "WORKS_AT" in prompt
    assert provider.last_request.params.get("temperature") == 0


async def test_low_scores_recommend(graph_context: Any, model_cfg: Any) -> None:
    provider = _FakeProvider('{"realism": 1, "coherence": 2, "rationale": "off"}')
    score = await SemanticPlausibilityGraphJudge(provider, model_cfg).evaluate(  # type: ignore[arg-type]
        graph_context(_graph())
    )
    assert score.recommendations


async def test_no_edges_not_applicable(
    dataset_from_nx: Any, graph_context: Any, model_cfg: Any
) -> None:
    import networkx as nx  # type: ignore[import-untyped]

    graph = dataset_from_nx(nx.empty_graph(3))  # nodes, no edges
    with pytest.raises(JudgeNotApplicable):
        await SemanticPlausibilityGraphJudge(_FakeProvider("{}"), model_cfg).evaluate(  # type: ignore[arg-type]
            graph_context(graph)
        )


async def test_malformed_output_raises(graph_context: Any, model_cfg: Any) -> None:
    with pytest.raises(SemanticPlausibilityError):
        await SemanticPlausibilityGraphJudge(_FakeProvider("not json"), model_cfg).evaluate(  # type: ignore[arg-type]
            graph_context(_graph())
        )
