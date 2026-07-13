"""Templated multi-hop question generation with graph-derived gold answers.

For each sampled :class:`QAPath` a template family is chosen (seeded) and a
natural-language question is built. **The answer is always computed from the
graph traversal, never from the LLM.** An optional ``LLMProvider`` only rewrites
the question's surface phrasing; if its output is empty (or no provider is
given) the template phrasing is used verbatim. This confines the LLM to
cosmetics and guarantees every gold answer is verifiably grounded in the graph.

Four template families (all covered when ``num_questions >= 4``):

- ``chained_relation`` — follow the path's relation chain to the terminal node.
- ``aggregation_count`` — count a node's neighbors along a relation type.
- ``existence_negation`` — whether a node has a given relation (yes / no).
- ``comparison`` — which of two path endpoints has more relationships.
"""

from __future__ import annotations

import numpy as np
from anodyne_core.models import LLMRequest, Message, ModelConfig
from anodyne_core.ports import LLMProvider

from anodyne_graph.graphrag.models import GraphQAFixture, GraphQAItem, QAPath
from anodyne_graph.graphrag.pathfinder import MAX_HOPS, MIN_HOPS, sample_paths
from anodyne_graph.models import Edge, GraphDataset, Node

CHAINED_RELATION = "chained_relation"
AGGREGATION_COUNT = "aggregation_count"
EXISTENCE_NEGATION = "existence_negation"
COMPARISON = "comparison"
FAMILIES: tuple[str, ...] = (
    CHAINED_RELATION,
    AGGREGATION_COUNT,
    EXISTENCE_NEGATION,
    COMPARISON,
)

_NAME_KEYS = ("name", "full_name", "title", "label", "username")


def _node_label(node: Node) -> str:
    """A human-facing label for a node: a name-like property, else its id."""
    for key in _NAME_KEYS:
        value = node.properties.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return node.id


def _rel_phrase(edge_type: str) -> str:
    """Humanize an edge-type name (``FOUNDED_BY`` -> ``founded by``)."""
    return edge_type.replace("_", " ").replace("-", " ").strip().lower() or edge_type


def _difficulty(hop_count: int) -> str:
    return {2: "easy", 3: "medium"}.get(hop_count, "hard")


class GraphRAGGenerator:
    """Synthesizes a :class:`GraphQAFixture` from a ``GraphDataset``.

    ``provider``/``model_config`` are optional; when both are supplied the LLM
    refines question phrasing only. Answers and gold paths are always derived
    from the graph.
    """

    def __init__(
        self,
        provider: LLMProvider | None = None,
        model_config: ModelConfig | None = None,
    ) -> None:
        if provider is not None and model_config is None:
            raise ValueError("model_config is required when a provider is given")
        self._provider = provider
        self._cfg = model_config

    async def generate(
        self,
        dataset: GraphDataset,
        dataset_version_id: str,
        *,
        num_questions: int,
        seed: int,
        min_hops: int = MIN_HOPS,
        max_hops: int = MAX_HOPS,
    ) -> GraphQAFixture:
        paths = sample_paths(
            dataset, count=num_questions, seed=seed, min_hops=min_hops, max_hops=max_hops
        )
        rng = np.random.default_rng([seed, 1])
        # Seeded family order guarantees all four families appear (count >= 4)
        # while staying reproducible.
        order = [FAMILIES[i] for i in rng.permutation(len(FAMILIES))]

        nodes_by_id = {n.id: n for n in dataset.nodes}
        edges_by_id = {e.id: e for e in dataset.edges}
        neighbors = _typed_neighbor_index(dataset)
        degree = _degree_index(dataset)
        rel_types = sorted({e.type for e in dataset.edges})

        items: list[GraphQAItem] = []
        for i, path in enumerate(paths):
            family = order[i % len(order)]
            question, answer, answer_nodes = self._build(
                family, path, nodes_by_id, edges_by_id, neighbors, degree, rel_types, rng
            )
            phrased = await self._refine(question, seed, i)
            items.append(
                GraphQAItem(
                    question=phrased,
                    answer=answer,
                    answer_node_ids=answer_nodes,
                    gold_path=path,
                    hop_count=path.hop_count,
                    question_type=family,
                    difficulty=_difficulty(path.hop_count),
                )
            )

        type_counts: dict[str, int] = {}
        for item in items:
            type_counts[item.question_type] = type_counts.get(item.question_type, 0) + 1
        metadata = {
            "seed": seed,
            "num_questions": len(items),
            "min_hops": min_hops,
            "max_hops": max_hops,
            "question_type_counts": dict(sorted(type_counts.items())),
            "llm_refined": self._provider is not None,
        }
        return GraphQAFixture(dataset_version_id=dataset_version_id, items=items, metadata=metadata)

    def _build(
        self,
        family: str,
        path: QAPath,
        nodes_by_id: dict[str, Node],
        edges_by_id: dict[str, Edge],
        neighbors: dict[tuple[str, str], list[str]],
        degree: dict[str, int],
        rel_types: list[str],
        rng: np.random.Generator,
    ) -> tuple[str, str, list[str]]:
        """Return ``(template_question, gold_answer, answer_node_ids)``.

        The gold answer is always computed here from graph structure.
        """
        start = nodes_by_id[path.start_node_id]
        terminal = nodes_by_id[path.terminal_node_id]

        if family == CHAINED_RELATION:
            rels = [_rel_phrase(edges_by_id[eid].type) for eid in path.edge_ids]
            chain = ", then ".join(rels)
            question = (
                f"Starting from {_node_label(start)}, which entity do you reach "
                f"by following the {chain} relationship?"
            )
            return question, _node_label(terminal), [terminal.id]

        if family == AGGREGATION_COUNT:
            rel_type = edges_by_id[path.edge_ids[0]].type
            nbrs = neighbors.get((start.id, rel_type), [])
            question = (
                f"How many {_rel_phrase(rel_type)} relationships does {_node_label(start)} have?"
            )
            return question, str(len(nbrs)), sorted(nbrs)

        if family == EXISTENCE_NEGATION:
            rel_type = self._existence_relation(start, edges_by_id, path, rel_types, neighbors, rng)
            # Ground the answer in the SAME undirected-aware neighbor index every
            # other family uses: a node "has" the relation iff that traversal
            # reaches at least one neighbor (so an endpoint on the *target* side
            # of an undirected edge counts). Never trust a source-only view.
            nbrs = neighbors.get((start.id, rel_type), [])
            exists = bool(nbrs)
            question = f"Does {_node_label(start)} have any {_rel_phrase(rel_type)} relationship?"
            answer = "Yes" if exists else "No"
            return question, answer, sorted(nbrs) if exists else []

        # COMPARISON
        d_start, d_term = degree.get(start.id, 0), degree.get(terminal.id, 0)
        question = (
            f"Which entity has more relationships: {_node_label(start)} or {_node_label(terminal)}?"
        )
        if d_start > d_term:
            return question, _node_label(start), [start.id]
        if d_term > d_start:
            return question, _node_label(terminal), [terminal.id]
        return (
            question,
            f"{_node_label(start)} and {_node_label(terminal)} have an equal number "
            f"of relationships.",
            sorted({start.id, terminal.id}),
        )

    @staticmethod
    def _existence_relation(
        start: Node,
        edges_by_id: dict[str, Edge],
        path: QAPath,
        rel_types: list[str],
        neighbors: dict[tuple[str, str], list[str]],
        rng: np.random.Generator,
    ) -> str:
        """Pick which relation type an existence question asks about.

        Sometimes a relation the start node genuinely has (leading to a "yes"),
        sometimes one it lacks (a "no"). Presence/absence is decided by the same
        undirected-aware `neighbors` index the caller grounds the answer with, so
        an endpoint on the *target* side of an undirected relation is correctly
        seen as having it. The caller computes the yes/no truth from `neighbors`.
        """
        present = edges_by_id[path.edge_ids[0]].type
        has = {rt for rt in rel_types if neighbors.get((start.id, rt))}
        absent = [rt for rt in rel_types if rt not in has]
        if absent and bool(rng.integers(0, 2)):
            return absent[int(rng.integers(0, len(absent)))]
        return present

    async def _refine(self, question: str, seed: int, index: int) -> str:
        """LLM surface rewrite only; empty output falls back to the template."""
        if self._provider is None or self._cfg is None:
            return question
        request = LLMRequest(
            model_config_id=self._cfg.id,
            messages=[
                Message(
                    role="system",
                    content=(
                        "Rewrite the user's question to sound more natural. Keep "
                        "the exact meaning, entities, and relationships. Return "
                        "only the rewritten question, no prose."
                    ),
                ),
                Message(role="user", content=question),
            ],
            params={"temperature": 0, "seed": seed + index},
        )
        response = await self._provider.complete(self._cfg, request)
        refined = response.content.strip()
        return refined or question


def _typed_neighbor_index(dataset: GraphDataset) -> dict[tuple[str, str], list[str]]:
    """(node_id, edge_type) -> neighbor ids reachable via that relation.

    Respects direction: directed edges count only from their source; undirected
    edge types count from either endpoint.
    """
    directed = {et.name: et.directed for et in dataset.ontology.edge_types}
    index: dict[tuple[str, str], list[str]] = {}
    for edge in dataset.edges:
        index.setdefault((edge.source, edge.type), []).append(edge.target)
        if not directed.get(edge.type, True):
            index.setdefault((edge.target, edge.type), []).append(edge.source)
    return index


def _degree_index(dataset: GraphDataset) -> dict[str, int]:
    """node_id -> number of incident edges (undirected degree)."""
    degree: dict[str, int] = {}
    for edge in dataset.edges:
        degree[edge.source] = degree.get(edge.source, 0) + 1
        degree[edge.target] = degree.get(edge.target, 0) + 1
    return degree
