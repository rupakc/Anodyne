from __future__ import annotations

from anodyne_graph.graphrag.models import GraphQAFixture, GraphQAItem, QAPath


def _path() -> QAPath:
    return QAPath(
        hops=[("Person:0", "KNOWS:0"), ("Person:1", "KNOWS:3")], terminal_node_id="Person:2"
    )


def test_qapath_derived_properties() -> None:
    p = _path()
    assert p.hop_count == 2
    assert p.edge_ids == ["KNOWS:0", "KNOWS:3"]
    assert p.node_ids == ["Person:0", "Person:1", "Person:2"]
    assert p.start_node_id == "Person:0"


def test_qapath_round_trips_through_json() -> None:
    p = _path()
    restored = QAPath.model_validate(p.model_dump(mode="json"))
    assert restored == p
    # tuples survive the list->tuple coercion
    assert restored.hops[0] == ("Person:0", "KNOWS:0")


def test_qa_item_and_fixture_shapes() -> None:
    item = GraphQAItem(
        question="Q?",
        answer="A",
        answer_node_ids=["Person:2"],
        gold_path=_path(),
        hop_count=2,
        question_type="chained_relation",
        difficulty="easy",
    )
    fixture = GraphQAFixture(
        dataset_version_id="v1",
        items=[item],
        metadata={"seed": 7},
    )
    assert fixture.items[0].gold_path.terminal_node_id == "Person:2"
    assert fixture.metadata["seed"] == 7
    # fixture round-trips
    assert GraphQAFixture.model_validate(fixture.model_dump(mode="json")) == fixture
