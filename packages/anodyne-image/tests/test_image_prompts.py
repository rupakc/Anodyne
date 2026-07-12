from __future__ import annotations

from uuid import uuid4

from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_image.prompts import ImagePromptBuilder


def _spec(
    description: str = "product photo",
    choices: list[str] | None = None,
    directives: dict[str, object] | None = None,
    target_rows: int = 100,
) -> DatasetSpec:
    fields = (
        [
            FieldSpec(
                name="label",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": choices},
            )
        ]
        if choices
        else []
    )
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description=description,
        modality=Modality.IMAGE,
        source="description",
        fields=fields,
        target_rows=target_rows,
        directives=directives or {},
    )


def test_deterministic_same_args() -> None:
    spec = _spec(choices=["cat", "dog"])
    a = ImagePromptBuilder().build(spec, 0, 20)
    b = ImagePromptBuilder().build(spec, 0, 20)
    assert [i.model_dump() for i in a] == [i.model_dump() for i in b]


def test_item_count_and_indices() -> None:
    spec = _spec(choices=["cat", "dog"])
    items = ImagePromptBuilder().build(spec, 10, 5)
    assert len(items) == 5
    assert [i.item_index for i in items] == [10, 11, 12, 13, 14]


def test_labels_rotate_through_choices() -> None:
    spec = _spec(choices=["cat", "dog", "bird"])
    items = ImagePromptBuilder().build(spec, 0, 6)
    assert [i.label for i in items] == ["cat", "dog", "bird", "cat", "dog", "bird"]


def test_label_rotation_consistent_across_shard_boundary() -> None:
    """The label for a given item_index must not depend on which shard produced it."""
    spec = _spec(choices=["cat", "dog", "bird"])
    whole = ImagePromptBuilder().build(spec, 0, 20)
    shard = ImagePromptBuilder().build(spec, 10, 5)
    for item in shard:
        assert item.label == whole[item.item_index].label
        assert item.prompt == whole[item.item_index].prompt


def test_no_choices_yields_no_label() -> None:
    spec = _spec(choices=None)
    items = ImagePromptBuilder().build(spec, 0, 3)
    assert all(i.label is None for i in items)
    assert all(i.prompt for i in items)


def test_directives_labels_fallback_used_when_no_field() -> None:
    spec = _spec(choices=None, directives={"labels": ["red", "blue"]})
    items = ImagePromptBuilder().build(spec, 0, 4)
    assert [i.label for i in items] == ["red", "blue", "red", "blue"]


def test_prompt_incorporates_description_and_directives() -> None:
    spec = _spec(
        description="a shoe",
        choices=["red"],
        directives={
            "bias": "worn outdoors",
            "use_case": "e-commerce",
            "edge_case": "torn laces",
            "style": "studio lighting",
        },
    )
    item = ImagePromptBuilder().build(spec, 0, 1)[0]
    assert "a shoe" in item.prompt
    assert "red" in item.prompt
    assert "worn outdoors" in item.prompt
    assert "e-commerce" in item.prompt
    assert "torn laces" in item.prompt
    assert "studio lighting" in item.prompt


def test_empty_description_and_no_label_still_produces_prompt() -> None:
    spec = _spec(description="", choices=None)
    item = ImagePromptBuilder().build(spec, 0, 1)[0]
    assert item.prompt
