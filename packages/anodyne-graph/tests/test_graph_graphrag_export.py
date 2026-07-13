from __future__ import annotations

import json

from anodyne_graph.graphrag.export import fixture_to_jsonl, graphrag_manifest
from anodyne_graph.graphrag.models import GraphQAFixture, GraphQAItem, QAPath


def _fixture() -> GraphQAFixture:
    items = [
        GraphQAItem(
            question="Who does P0 know two hops away?",
            answer="P2",
            answer_node_ids=["Person:2"],
            gold_path=QAPath(
                hops=[("Person:0", "K0"), ("Person:1", "K1")],
                terminal_node_id="Person:2",
            ),
            hop_count=2,
            question_type="chained_relation",
            difficulty="easy",
        ),
        GraphQAItem(
            question="How many people does P0 know?",
            answer="3",
            answer_node_ids=["Person:1", "Person:4", "Person:5"],
            gold_path=QAPath(hops=[("Person:0", "K0")], terminal_node_id="Person:1"),
            hop_count=1,
            question_type="aggregation_count",
            difficulty="easy",
        ),
    ]
    return GraphQAFixture(dataset_version_id="v1", items=items, metadata={"seed": 7})


def test_jsonl_schema_one_object_per_item() -> None:
    fixture = _fixture()
    lines = fixture_to_jsonl(fixture).decode("utf-8").splitlines()
    assert len(lines) == len(fixture.items)
    for line, item in zip(lines, fixture.items, strict=True):
        obj = json.loads(line)
        assert set(obj) == {
            "question",
            "answer",
            "answer_node_ids",
            "hop_count",
            "question_type",
            "difficulty",
            "gold_path",
        }
        assert obj["answer"] == item.answer
        assert set(obj["gold_path"]) == {"hops", "node_ids", "edge_ids", "terminal_node_id"}
        assert obj["gold_path"]["edge_ids"] == item.gold_path.edge_ids
        assert obj["gold_path"]["node_ids"] == item.gold_path.node_ids


def test_jsonl_is_byte_identical_for_same_fixture() -> None:
    fixture = _fixture()
    assert fixture_to_jsonl(fixture) == fixture_to_jsonl(fixture)


def test_empty_fixture_serializes_to_empty_bytes() -> None:
    empty = GraphQAFixture(dataset_version_id="v1")
    assert fixture_to_jsonl(empty) == b""


def test_manifest_describes_fixture() -> None:
    fixture = _fixture()
    manifest = graphrag_manifest(fixture)
    assert manifest["dataset_version_id"] == "v1"
    assert manifest["item_count"] == 2
    assert manifest["seed"] == 7
    assert sum(manifest["question_type_counts"].values()) == 2
    assert manifest["question_type_counts"]["chained_relation"] == 1
    assert sum(manifest["hop_count_distribution"].values()) == 2
