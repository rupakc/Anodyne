from __future__ import annotations

from enum import StrEnum

from anodyne_dataset.models import FieldSpec, Modality, SemanticType
from anodyne_text.shapes import TextShape, detect_shape


class TaskType(StrEnum):
    TEXT_CLASSIFICATION = "text_classification"
    QA = "qa"
    SUMMARIZATION = "summarization"
    CHAT = "chat"
    TABULAR_CLASSIFICATION = "tabular_classification"
    REGRESSION = "regression"
    IMAGE_CLASSIFICATION = "image_classification"
    IMAGE_GENERATION = "image_generation"
    AUDIO_CLASSIFICATION = "audio_classification"
    SPEECH_SYNTHESIS = "speech_synthesis"
    TEXT_TO_VIDEO = "text_to_video"
    GRAPH_QA = "graph_qa"
    GENERIC = "generic"


_TEXT_SHAPE_MAP: dict[TextShape, TaskType] = {
    TextShape.CLASSIFICATION: TaskType.TEXT_CLASSIFICATION,
    TextShape.QA: TaskType.QA,
    TextShape.SUMMARIZATION: TaskType.SUMMARIZATION,
    TextShape.CHAT: TaskType.CHAT,
    TextShape.GENERIC: TaskType.GENERIC,
}


def detect_task(
    modality: Modality,
    columns: list[str],
    *,
    target_field: str | None = None,
    target_is_numeric: bool = False,
) -> TaskType:
    """Infer the task-class from modality + available columns (+ tabular target)."""
    names = set(columns)
    if modality == Modality.TEXT:
        # detect_shape keys on field names only; wrap columns as throwaway FieldSpecs.
        fields = [FieldSpec(name=c, semantic_type=SemanticType.TEXT) for c in columns]
        return _TEXT_SHAPE_MAP[detect_shape(fields)]
    if modality == Modality.TABULAR:
        if target_field is None:
            return TaskType.GENERIC
        return TaskType.REGRESSION if target_is_numeric else TaskType.TABULAR_CLASSIFICATION
    if modality == Modality.IMAGE:
        return TaskType.IMAGE_CLASSIFICATION if "label" in names else TaskType.IMAGE_GENERATION
    if modality == Modality.AUDIO:
        return TaskType.AUDIO_CLASSIFICATION if "label" in names else TaskType.SPEECH_SYNTHESIS
    if modality == Modality.VIDEO:
        return TaskType.TEXT_TO_VIDEO
    return TaskType.GENERIC
