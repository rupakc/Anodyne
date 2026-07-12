from __future__ import annotations

from enum import StrEnum

from anodyne_dataset.models import FieldSpec


class TextShape(StrEnum):
    """Common text-corpus row shapes, inferred from `DatasetSpec.fields` names.

    No new `DatasetSpec`/`FieldSpec` field is introduced for this: keeping shape
    detection purely name-driven avoids touching the shared `anodyne-dataset`
    domain model (and any risk of colliding with a parallel tabular-modality
    effort that may also depend on it unchanged).
    """

    CLASSIFICATION = "classification"
    QA = "qa"
    SUMMARIZATION = "summarization"
    CHAT = "chat"
    GENERIC = "generic"


# Each shape's required field-name subset, most-specific first. `detect_shape`
# matches on subset (not exact set) so extra columns (e.g. a "source" field
# alongside "text"/"label") don't fall through to GENERIC.
_SHAPE_FIELDS: dict[TextShape, tuple[frozenset[str], str]] = {
    TextShape.CLASSIFICATION: (frozenset({"text", "label"}), "text"),
    TextShape.QA: (frozenset({"question", "answer"}), "question"),
    TextShape.SUMMARIZATION: (frozenset({"document", "summary"}), "document"),
    TextShape.CHAT: (frozenset({"instruction", "response"}), "instruction"),
}
_CHAT_MESSAGES_FIELD = "messages"


def detect_shape(fields: list[FieldSpec]) -> TextShape:
    """Infer the text-corpus shape from field names.

    Matches the most specific known shape whose required field names are all
    present; a lone "messages" field is also recognized as CHAT (a
    JSON-encoded multi-turn conversation in a single column). Anything else
    falls back to GENERIC (a free-form structured row matching exactly the
    given field names).
    """
    names = {f.name for f in fields}
    for shape, (required, _primary) in _SHAPE_FIELDS.items():
        if required <= names:
            return shape
    if _CHAT_MESSAGES_FIELD in names:
        return TextShape.CHAT
    return TextShape.GENERIC


def primary_field(shape: TextShape, fields: list[FieldSpec]) -> str:
    """The field whose value quality-filtering/deduplication key on for `shape`."""
    names = {f.name for f in fields}
    if shape is TextShape.CHAT and _CHAT_MESSAGES_FIELD in names and "instruction" not in names:
        return _CHAT_MESSAGES_FIELD
    entry = _SHAPE_FIELDS.get(shape)
    if entry is not None:
        return entry[1]
    # GENERIC (or CHAT without a recognized field set, defensively): key on the
    # first declared field.
    return fields[0].name
