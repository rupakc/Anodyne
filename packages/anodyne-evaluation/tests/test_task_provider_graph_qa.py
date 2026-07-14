from __future__ import annotations

import json

# Importing the package registers every provider (graph_qa included) as a side
# effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import provider_for
from anodyne_graph.graphrag.models import GraphQAItem, QAPath
from anodyne_graph.models import Edge, EdgeType, GraphDataset, GraphOntology, Node, NodeType

# --- fixture graph: n1 --e1-- n2 --e2-- n3 -----------------------------------


def _graph() -> GraphDataset:
    ontology = GraphOntology(
        node_types=[NodeType(name="Item")],
        edge_types=[EdgeType(name="LINK", source_type="Item", target_type="Item")],
    )
    nodes = [Node(id="n1", type="Item"), Node(id="n2", type="Item"), Node(id="n3", type="Item")]
    edges = [
        Edge(id="e1", type="LINK", source="n1", target="n2"),
        Edge(id="e2", type="LINK", source="n2", target="n3"),
    ]
    return GraphDataset(ontology=ontology, nodes=nodes, edges=edges)


def _resolvable_item() -> GraphQAItem:
    """Gold path fully resolves and re-traverses n1 -> n2 -> n3 == terminal."""
    return GraphQAItem(
        question="How is n1 connected to n3?",
        answer="via n2",
        answer_node_ids=["n3"],
        gold_path=QAPath(hops=[("n1", "e1"), ("n2", "e2")], terminal_node_id="n3"),
        hop_count=2,
        question_type="multi_hop",
        difficulty="medium",
    )


def _unresolvable_item() -> GraphQAItem:
    """Terminal node id `n9` doesn't exist on the graph -- unanswerable, and the
    retraversal (n1 -> n2 via e1) never reaches it -- incorrect. `answer_node_ids`
    is empty, so it also counts as ungrounded."""
    return GraphQAItem(
        question="What lies beyond n3?",
        answer="n9",
        answer_node_ids=[],
        gold_path=QAPath(hops=[("n1", "e1")], terminal_node_id="n9"),
        hop_count=1,
        question_type="multi_hop",
        difficulty="easy",
    )


_ALL_SELECTED = frozenset(
    {"answerable_rate", "multi_hop_correctness", "answer_groundedness", "question_clarity"}
)


class _FakeGraphQAProvider:
    def __init__(self) -> None:
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(
            content=json.dumps({"question_clarity": 4}),
            usage=Usage(total_tokens=1),
        )

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


class _BadContentProvider:
    def __init__(self, content: str) -> None:
        self._content = content
        self.last_request: LLMRequest | None = None

    async def complete(self, config, request):  # type: ignore[no-untyped-def]
        self.last_request = request
        return LLMResponse(content=self._content, usage=Usage(total_tokens=1))

    def stream(self, config, request): ...  # type: ignore[no-untyped-def]


def _ctx(**kwargs: object) -> EvaluationContext:
    defaults: dict[str, object] = dict(
        subject=pd.DataFrame(),
        task_type=TaskType.GRAPH_QA,
        subject_graph=_graph(),
        graph_qa_items=[_resolvable_item(), _unresolvable_item()],
        sample_rows=20,
    )
    defaults.update(kwargs)
    return EvaluationContext(**defaults)  # type: ignore[arg-type]


async def test_graph_qa_provider_scores_all_metrics(model_cfg: ModelConfig) -> None:
    llm = _FakeGraphQAProvider()
    ctx = _ctx()
    prov = provider_for(TaskType.GRAPH_QA)
    assert prov is not None
    score = await prov.score(ctx, llm, model_cfg, selected=_ALL_SELECTED)  # type: ignore[arg-type]
    assert score.dimension.value == "task_quality"
    assert score.metrics["answerable_rate"] == pytest.approx(0.5)
    assert score.metrics["multi_hop_correctness"] == pytest.approx(0.5)
    assert score.metrics["answer_groundedness"] == pytest.approx(0.5)
    assert score.metrics["question_clarity"] == pytest.approx(0.8)
    expected_score = sum(score.metrics[k] for k in _ALL_SELECTED) / len(_ALL_SELECTED)
    assert score.score == pytest.approx(expected_score)
    assert llm.last_request is not None
    assert llm.last_request.params.get("temperature") == 0


async def test_graph_qa_provider_skips_llm_when_not_selected(model_cfg: ModelConfig) -> None:
    class _ExplodingProvider:
        async def complete(self, config, request):  # type: ignore[no-untyped-def]
            raise AssertionError("LLM should not be called when no LLM metric is selected")

        def stream(self, config, request): ...  # type: ignore[no-untyped-def]

    ctx = _ctx()
    prov = provider_for(TaskType.GRAPH_QA)
    assert prov is not None
    score = await prov.score(
        ctx,
        _ExplodingProvider(),  # type: ignore[arg-type]
        model_cfg,
        selected=frozenset({"answerable_rate", "multi_hop_correctness", "answer_groundedness"}),
    )
    assert set(score.metrics) == {"answerable_rate", "multi_hop_correctness", "answer_groundedness"}


async def test_graph_qa_provider_no_items_raises(model_cfg: ModelConfig) -> None:
    ctx = _ctx(graph_qa_items=[])
    prov = provider_for(TaskType.GRAPH_QA)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _FakeGraphQAProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"answerable_rate"}),
        )


async def test_graph_qa_provider_no_subject_graph_raises(model_cfg: ModelConfig) -> None:
    ctx = _ctx(subject_graph=None)
    prov = provider_for(TaskType.GRAPH_QA)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _FakeGraphQAProvider(),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"answerable_rate"}),
        )


async def test_graph_qa_provider_llm_parse_errors(model_cfg: ModelConfig) -> None:
    ctx = _ctx()
    prov = provider_for(TaskType.GRAPH_QA)
    assert prov is not None
    with pytest.raises(TaskMetricError):
        await prov.score(
            ctx,
            _BadContentProvider("not json"),  # type: ignore[arg-type]
            model_cfg,
            selected=frozenset({"question_clarity"}),
        )
