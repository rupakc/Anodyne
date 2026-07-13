"""Domain models for GraphRAG multi-hop QA fixtures.

Pure Pydantic + stdlib (no adapter imports). A ``QAPath`` is the *gold
supporting path* through the graph that a question is grounded in: an ordered
list of ``(node_id, edge_id)`` hops plus the terminal node reached. Each hop's
``node_id`` is the node the traversal is *at* when it crosses ``edge_id`` to the
next node; ``node_ids`` therefore has exactly ``hop_count + 1`` entries.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QAPath(BaseModel):
    """The gold supporting path a QA item is grounded in.

    ``hops`` is an ordered list of ``(node_id, edge_id)`` pairs; ``node_id`` is
    the source end of the hop (the node traversal is at) and ``edge_id`` the
    real graph edge crossed. ``terminal_node_id`` is the node reached after the
    final hop.
    """

    hops: list[tuple[str, str]] = Field(default_factory=list)
    terminal_node_id: str

    @property
    def hop_count(self) -> int:
        return len(self.hops)

    @property
    def edge_ids(self) -> list[str]:
        return [edge_id for _, edge_id in self.hops]

    @property
    def node_ids(self) -> list[str]:
        """All node ids on the path, in order (``hop_count + 1`` entries)."""
        return [node_id for node_id, _ in self.hops] + [self.terminal_node_id]

    @property
    def start_node_id(self) -> str:
        """The first node on the path (its terminal if there are no hops)."""
        return self.hops[0][0] if self.hops else self.terminal_node_id


class GraphQAItem(BaseModel):
    """One multi-hop QA item: question surface form + graph-derived gold answer.

    ``answer`` is always computed from the graph (never the LLM);
    ``answer_node_ids`` are the graph nodes that ground the answer; ``gold_path``
    is the supporting path; ``question_type`` is one of the four template
    families; ``difficulty`` is derived from ``hop_count``.
    """

    question: str
    answer: str
    answer_node_ids: list[str] = Field(default_factory=list)
    gold_path: QAPath
    hop_count: int
    question_type: str
    difficulty: str


class GraphQAFixture(BaseModel):
    """A GraphRAG evaluation fixture: the QA items + provenance metadata.

    ``metadata`` records at least ``seed`` (so the fixture is reproducible) plus
    the hop bounds and per-type counts.
    """

    dataset_version_id: str
    items: list[GraphQAItem] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
