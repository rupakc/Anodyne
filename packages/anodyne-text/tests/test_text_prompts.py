from uuid import uuid4

from anodyne_dataset.models import DatasetSpec, FieldSpec, Modality, SemanticType
from anodyne_text.prompts import build_batch_prompt
from anodyne_text.shapes import TextShape


def _spec(fields: list[FieldSpec], directives: dict[str, object] | None = None) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=uuid4(),
        name="d",
        description="customer support tickets",
        modality=Modality.TEXT,
        source="description",
        fields=fields,
        target_rows=100,
        directives=directives or {},
    )


def test_prompt_has_system_and_user_message() -> None:
    spec = _spec(
        [
            FieldSpec(name="text", semantic_type=SemanticType.TEXT),
            FieldSpec(name="label", semantic_type=SemanticType.CATEGORICAL),
        ]
    )
    messages = build_batch_prompt(
        spec, TextShape.CLASSIFICATION, batch_size=10, seed=1, batch_index=0
    )
    assert [m.role for m in messages] == ["system", "user"]


def test_prompt_mentions_all_field_names() -> None:
    spec = _spec(
        [
            FieldSpec(name="question", semantic_type=SemanticType.TEXT),
            FieldSpec(name="answer", semantic_type=SemanticType.TEXT),
        ]
    )
    messages = build_batch_prompt(spec, TextShape.QA, batch_size=5, seed=1, batch_index=0)
    system = messages[0].content
    assert "question" in system and "answer" in system


def test_prompt_folds_in_directives() -> None:
    spec = _spec(
        [
            FieldSpec(name="text", semantic_type=SemanticType.TEXT),
            FieldSpec(name="label", semantic_type=SemanticType.CATEGORICAL),
        ],
        directives={"topic": "billing disputes", "tone": "frustrated"},
    )
    messages = build_batch_prompt(
        spec, TextShape.CLASSIFICATION, batch_size=10, seed=1, batch_index=0
    )
    system = messages[0].content
    assert "billing disputes" in system
    assert "frustrated" in system


def test_prompt_user_message_mentions_batch_size() -> None:
    spec = _spec([FieldSpec(name="text", semantic_type=SemanticType.TEXT)])
    messages = build_batch_prompt(spec, TextShape.GENERIC, batch_size=17, seed=1, batch_index=0)
    assert "17" in messages[1].content
