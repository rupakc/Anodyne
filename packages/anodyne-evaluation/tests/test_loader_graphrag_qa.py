"""Tests for `load_graphrag_qa`: parsing a GraphRAG QA fixture's bytes into
`GraphQAItem`s, mirroring `load_manifest`'s dict-or-bare-list acceptance.
"""

from __future__ import annotations

import json

from anodyne_evaluation.loader import load_graphrag_qa
from anodyne_graph.graphrag.models import GraphQAItem


def _item(question: str = "who is connected to n0?") -> dict[str, object]:
    return {
        "question": question,
        "answer": "n1",
        "answer_node_ids": ["n1"],
        "gold_path": {"hops": [["n0", "e0"]], "terminal_node_id": "n1"},
        "hop_count": 1,
        "question_type": "single_hop",
        "difficulty": "easy",
    }


def test_loads_bare_list_of_items() -> None:
    data = json.dumps([_item("q1"), _item("q2")]).encode()
    items = load_graphrag_qa(data)
    assert [type(i) for i in items] == [GraphQAItem, GraphQAItem]
    assert [i.question for i in items] == ["q1", "q2"]


def test_loads_dict_form_with_items_key() -> None:
    data = json.dumps({"items": [_item("q1")]}).encode()
    items = load_graphrag_qa(data)
    assert len(items) == 1
    assert items[0].question == "q1"
    assert items[0].gold_path.terminal_node_id == "n1"


def test_empty_items_returns_empty_list() -> None:
    data = json.dumps({"items": []}).encode()
    assert load_graphrag_qa(data) == []
