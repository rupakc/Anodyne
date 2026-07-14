from __future__ import annotations

import json

# Importing the package registers every provider (graph_qa included) as a side
# effect of import, mirroring the `task_metrics/__init__.py` contract.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
import pandas as pd  # type: ignore[import-untyped]
import pytest
from anodyne_core.models import LLMRequest, LLMResponse, ModelConfig, Usage
from anodyne_evaluation.judges.task_metrics.base import TaskMetricError
from anodyne_evaluation.judges.task_metrics.graph_qa import (
    _build_indices,
    _multi_hop_correctness,
    _sample_questions,
)
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


# --- _sample_questions: seeded, reproducible subset selection ---------------


def _items_with_questions(n: int) -> list[GraphQAItem]:
    return [
        GraphQAItem(
            question=f"q{i}",
            answer="a",
            answer_node_ids=["n1"],
            gold_path=QAPath(hops=[], terminal_node_id="n1"),
            hop_count=0,
            question_type="multi_hop",
            difficulty="easy",
        )
        for i in range(n)
    ]


def test_sample_questions_seeded_reproducible_and_sized() -> None:
    items = _items_with_questions(5)
    ctx = _ctx(graph_qa_items=items, sample_rows=3, seed=42)

    first = _sample_questions(items, ctx)
    second = _sample_questions(items, ctx)

    # Same seed -> identical subset (order included), and the pinned selection
    # random.Random(42).sample(range(5), k=3) actually produces.
    assert first == second == ["q0", "q4", "q2"]
    assert len(first) == min(ctx.sample_rows, len(items))


def test_sample_questions_different_seed_differs() -> None:
    items = _items_with_questions(5)
    ctx_a = _ctx(graph_qa_items=items, sample_rows=3, seed=42)
    ctx_b = _ctx(graph_qa_items=items, sample_rows=3, seed=7)

    assert _sample_questions(items, ctx_a) != _sample_questions(items, ctx_b)


def test_sample_questions_caps_at_available_items() -> None:
    items = _items_with_questions(2)
    ctx = _ctx(graph_qa_items=items, sample_rows=20, seed=0)

    result = _sample_questions(items, ctx)

    assert len(result) == min(ctx.sample_rows, len(items)) == 2
    assert set(result) == {"q0", "q1"}


# --- _path_retraverses / _other_end: undirected + broken-traversal coverage -


def _linear_graph() -> GraphDataset:
    """n1 --e1(source=n1,target=n2)-- n2, matching `_graph()` above."""
    return _graph()


def test_multi_hop_correctness_target_side_hop_is_correct() -> None:
    """Edge e1 is stored as source=n1, target=n2, but the gold path's hop
    starts at n2 and crosses e1 back to n1 -- i.e. the path traverses the edge
    from its `target` side. `_other_end` must resolve n2 -> n1 by falling
    through to the `edge.target == node_id` branch, not just `edge.source`."""
    graph = _linear_graph()
    _, edges_by_id = _build_indices(graph)
    item = GraphQAItem(
        question="How is n2 connected to n1?",
        answer="via e1",
        answer_node_ids=["n1"],
        gold_path=QAPath(hops=[("n2", "e1")], terminal_node_id="n1"),
        hop_count=1,
        question_type="multi_hop",
        difficulty="easy",
    )
    assert _multi_hop_correctness([item], edges_by_id) == pytest.approx(1.0)


def test_multi_hop_correctness_missing_edge_id_counts_incorrect_no_raise() -> None:
    graph = _linear_graph()
    _, edges_by_id = _build_indices(graph)
    item = GraphQAItem(
        question="What connects n1 to n3 via a ghost edge?",
        answer="unknown",
        answer_node_ids=[],
        gold_path=QAPath(hops=[("n1", "e_missing")], terminal_node_id="n2"),
        hop_count=1,
        question_type="multi_hop",
        difficulty="easy",
    )
    assert _multi_hop_correctness([item], edges_by_id) == pytest.approx(0.0)


def test_multi_hop_correctness_disconnected_hop_counts_incorrect_no_raise() -> None:
    """The hop names e2 (a real edge, n2--n3) but claims to start the walk from
    n1 -- e2 doesn't touch n1 on either end, so `_other_end` returns `None` and
    the walk fails without raising."""
    graph = _linear_graph()
    _, edges_by_id = _build_indices(graph)
    item = GraphQAItem(
        question="Does e2 connect n1 to n3?",
        answer="no",
        answer_node_ids=[],
        gold_path=QAPath(hops=[("n1", "e2")], terminal_node_id="n3"),
        hop_count=1,
        question_type="multi_hop",
        difficulty="easy",
    )
    assert _multi_hop_correctness([item], edges_by_id) == pytest.approx(0.0)
