"""`TaskMetricProvider` for `TaskType.GRAPH_QA`.

Consumes the GRAPH artifacts on the context -- `ctx.subject_graph` (the
generated `GraphDataset`) and `ctx.graph_qa_items` (the multi-hop
`GraphQAItem` fixtures, sub-system GraphRAG) -- never the (empty) DataFrame
`ctx.subject` a graph run carries. Three metrics are intrinsic, pure graph
checks against each item's `gold_path`/`answer_node_ids`; only
`question_clarity` calls the LLM, and it judges question surface forms alone
(no graph is shown to the model).

Hop resolution is undirected-aware, mirroring how `anodyne_evaluation.graph_judges`
treats the LPG as a simple undirected graph for its structural metrics: a hop
`(node_id, edge_id)` is crossed by finding the edge's *other* endpoint relative
to wherever the retraversal currently stands, regardless of which end the edge
declares as `source`/`target`.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_evaluation.judges.task_metrics.base import (
    TaskMetricError,
    mean_contribution,
    strip_json,
)
from anodyne_evaluation.models import EvalDimension, ExpertScore
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, register_provider

if TYPE_CHECKING:
    # `anodyne_graph` ships no `py.typed` marker; mirror `graph_judges/base.py` and
    # `ports.py` by importing its types only for annotations (lazily evaluated as
    # strings under `from __future__ import annotations`, never at runtime), which
    # sidesteps the untyped-import error without adding a runtime adapter import.
    from anodyne_graph.graphrag.models import GraphQAItem, QAPath
    from anodyne_graph.models import Edge, GraphDataset

_QUESTION_CLARITY_SYSTEM = (
    "You are rating a batch of multi-hop graph question-answering questions for "
    "clarity, considering all questions below together as a single batch. Rate ONE "
    "criterion, an INTEGER from 1 (poor) to 5 (excellent): question_clarity (questions "
    "are clear, well-formed, and unambiguous). Return ONLY JSON: "
    '{"question_clarity": int}. No prose outside the JSON.'
)


# --- graph indices + hop resolution -----------------------------------------


def _build_indices(graph: GraphDataset) -> tuple[set[str], dict[str, Edge]]:
    node_ids = {n.id for n in graph.nodes}
    edges_by_id = {e.id: e for e in graph.edges}
    return node_ids, edges_by_id


def _other_end(edge: Edge, node_id: str) -> str | None:
    """The neighbor reached by crossing `edge` from `node_id`, treating the edge as
    undirected -- `None` if `node_id` isn't one of the edge's two endpoints."""
    if edge.source == node_id:
        return edge.target
    if edge.target == node_id:
        return edge.source
    return None


def _path_resolves(path: QAPath, node_ids: set[str], edges_by_id: dict[str, Edge]) -> bool:
    """`True` iff every edge/node id the path references actually exists on the
    graph (no traversal is attempted -- pure existence check)."""
    return all(eid in edges_by_id for eid in path.edge_ids) and all(
        nid in node_ids for nid in path.node_ids
    )


def _path_retraverses(path: QAPath, edges_by_id: dict[str, Edge]) -> bool:
    """Re-walk `path` from its `start_node_id`, crossing each hop's edge
    undirected-aware; `True` iff the walk lands on `terminal_node_id`. Any
    missing edge or a hop that doesn't connect to the current node fails the
    walk (returns `False`) rather than raising."""
    current = path.start_node_id
    for _node_id, edge_id in path.hops:
        edge = edges_by_id.get(edge_id)
        if edge is None:
            return False
        nxt = _other_end(edge, current)
        if nxt is None:
            return False
        current = nxt
    return current == path.terminal_node_id


def _answerable_rate(
    items: list[GraphQAItem], node_ids: set[str], edges_by_id: dict[str, Edge]
) -> float:
    if not items:
        return 0.0
    resolved = sum(1 for item in items if _path_resolves(item.gold_path, node_ids, edges_by_id))
    return resolved / len(items)


def _multi_hop_correctness(items: list[GraphQAItem], edges_by_id: dict[str, Edge]) -> float:
    if not items:
        return 0.0
    correct = sum(1 for item in items if _path_retraverses(item.gold_path, edges_by_id))
    return correct / len(items)


def _answer_groundedness(items: list[GraphQAItem], node_ids: set[str]) -> float:
    if not items:
        return 0.0
    grounded = sum(
        1
        for item in items
        if item.answer_node_ids and all(nid in node_ids for nid in item.answer_node_ids)
    )
    return grounded / len(items)


# --- question_clarity (LLM) --------------------------------------------------


def _sample_questions(items: list[GraphQAItem], ctx: EvaluationContext) -> list[str]:
    n = min(ctx.sample_rows, len(items))
    if n <= 0:
        return []
    ordered = sorted(range(len(items)), key=lambda i: items[i].question)
    return [items[i].question for i in ordered[:n]]


def _parse_question_clarity(raw: str) -> float:
    text = strip_json(raw)
    try:
        data = json.loads(text)
        v = data["question_clarity"]
        if isinstance(v, bool) or not isinstance(v, int) or not (1 <= v <= 5):
            raise ValueError(f"question_clarity must be an integer 1-5, got {v!r}")
        return v / 5.0
    except Exception as exc:  # json/validation errors -> domain error
        raise TaskMetricError(
            f"could not parse question_clarity rubric from model output: {exc}"
        ) from exc


class GraphQAProvider:
    """Standard metrics for the `graph_qa` task class."""

    task_type = TaskType.GRAPH_QA

    def metric_catalog(self) -> list[MetricSpec]:
        return [
            MetricSpec(
                key="answerable_rate",
                label="Answerable rate",
                description=(
                    "Fraction of QA items whose gold path fully resolves on the graph "
                    "(every edge id and node id it references exists)."
                ),
                requires_llm=False,
            ),
            MetricSpec(
                key="multi_hop_correctness",
                label="Multi-hop correctness",
                description=(
                    "Fraction of items whose gold path, re-traversed hop by hop from its "
                    "start node, reaches the recorded terminal node."
                ),
                requires_llm=False,
            ),
            MetricSpec(
                key="answer_groundedness",
                label="Answer groundedness",
                description=(
                    "Fraction of items whose answer_node_ids are non-empty and all present "
                    "as graph node ids."
                ),
                requires_llm=False,
            ),
            MetricSpec(
                key="question_clarity",
                label="Question clarity",
                description="LLM-judged 1-5 rating of question clarity/well-formedness.",
                requires_llm=True,
            ),
        ]

    async def score(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        *,
        selected: frozenset[str],
    ) -> ExpertScore:
        if not ctx.graph_qa_items or ctx.subject_graph is None:
            raise TaskMetricError(
                "graph_qa requires non-empty 'graph_qa_items' and a 'subject_graph'"
            )
        items: list[GraphQAItem] = ctx.graph_qa_items
        node_ids, edges_by_id = _build_indices(ctx.subject_graph)

        metrics: dict[str, float] = {}
        if "answerable_rate" in selected:
            metrics["answerable_rate"] = _answerable_rate(items, node_ids, edges_by_id)
        if "multi_hop_correctness" in selected:
            metrics["multi_hop_correctness"] = _multi_hop_correctness(items, edges_by_id)
        if "answer_groundedness" in selected:
            metrics["answer_groundedness"] = _answer_groundedness(items, node_ids)
        if "question_clarity" in selected:
            metrics["question_clarity"] = await self._clarity_oracle(
                ctx, provider, model_config, items
            )

        score = mean_contribution(metrics, selected)
        recs: list[str] = []
        if metrics.get("answerable_rate", 1.0) < 0.9:
            recs.append(
                "Some gold paths reference edge/node ids missing from the graph; "
                "review the GraphRAG fixture generator."
            )
        if metrics.get("multi_hop_correctness", 1.0) < 0.9:
            recs.append(
                "Some gold paths don't retraverse to their recorded terminal node; "
                "the fixture's hop sequence may be inconsistent with the graph."
            )
        if metrics.get("answer_groundedness", 1.0) < 0.9:
            recs.append(
                "Some answers have no (or only missing) supporting node ids; "
                "answers should always cite graph-grounded node ids."
            )
        return ExpertScore(
            dimension=EvalDimension.TASK_QUALITY,
            score=score,
            rationale=(
                f"Graph multi-hop QA standard metrics for task class '{self.task_type.value}'."
            ),
            metrics=metrics,
            recommendations=recs,
        )

    async def _clarity_oracle(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        items: list[GraphQAItem],
    ) -> float:
        questions = _sample_questions(items, ctx)
        if not questions:
            return 0.0
        lines = [f"{i}. {q}" for i, q in enumerate(questions, start=1)]
        req = LLMRequest(
            model_config_id=model_config.id,
            messages=[
                Message(role="system", content=_QUESTION_CLARITY_SYSTEM),
                Message(role="user", content="\n".join(lines)),
            ],
            # Deterministic scoring: temperature=0 so the same sample yields a reproducible verdict.
            params={"temperature": 0},
        )
        resp = await provider.complete(model_config, req)
        return _parse_question_clarity(resp.content)


register_provider(GraphQAProvider())
