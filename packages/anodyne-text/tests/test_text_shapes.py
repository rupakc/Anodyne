from anodyne_dataset.models import FieldSpec, SemanticType
from anodyne_text.shapes import TextShape, detect_shape, primary_field


def _fields(*names: str) -> list[FieldSpec]:
    return [FieldSpec(name=n, semantic_type=SemanticType.TEXT) for n in names]


def test_detect_classification() -> None:
    assert detect_shape(_fields("text", "label")) is TextShape.CLASSIFICATION


def test_detect_qa() -> None:
    assert detect_shape(_fields("question", "answer")) is TextShape.QA


def test_detect_summarization() -> None:
    assert detect_shape(_fields("document", "summary")) is TextShape.SUMMARIZATION


def test_detect_chat_instruction_response() -> None:
    assert detect_shape(_fields("instruction", "response")) is TextShape.CHAT


def test_detect_chat_messages() -> None:
    assert detect_shape(_fields("messages")) is TextShape.CHAT


def test_detect_generic_fallback() -> None:
    assert detect_shape(_fields("foo", "bar")) is TextShape.GENERIC


def test_detect_tolerates_extra_fields() -> None:
    # A superset of the classification field names still matches classification.
    assert detect_shape(_fields("text", "label", "source")) is TextShape.CLASSIFICATION


def test_primary_field_per_shape() -> None:
    assert primary_field(TextShape.CLASSIFICATION, _fields("text", "label")) == "text"
    assert primary_field(TextShape.QA, _fields("question", "answer")) == "question"
    assert primary_field(TextShape.SUMMARIZATION, _fields("document", "summary")) == "document"
    assert primary_field(TextShape.CHAT, _fields("instruction", "response")) == "instruction"
    assert primary_field(TextShape.CHAT, _fields("messages")) == "messages"


def test_primary_field_generic_uses_first_field() -> None:
    assert primary_field(TextShape.GENERIC, _fields("foo", "bar")) == "foo"
