from anodyne_dataset.models import Modality
from anodyne_evaluation.task import TaskType, detect_task


def test_text_shapes_map_to_task_types() -> None:
    assert detect_task(Modality.TEXT, ["text", "label"]) is TaskType.TEXT_CLASSIFICATION
    assert detect_task(Modality.TEXT, ["question", "answer"]) is TaskType.QA
    assert detect_task(Modality.TEXT, ["document", "summary"]) is TaskType.SUMMARIZATION
    assert detect_task(Modality.TEXT, ["instruction", "response"]) is TaskType.CHAT
    assert detect_task(Modality.TEXT, ["freeform"]) is TaskType.GENERIC


def test_tabular_task_from_target() -> None:
    assert (
        detect_task(Modality.TABULAR, ["a", "y"], target_field="y")
        is TaskType.TABULAR_CLASSIFICATION
    )
    assert (
        detect_task(Modality.TABULAR, ["a", "y"], target_field="y", target_is_numeric=True)
        is TaskType.REGRESSION
    )
    assert detect_task(Modality.TABULAR, ["a", "b"]) is TaskType.GENERIC


def test_media_tasks_from_label_presence() -> None:
    assert detect_task(Modality.IMAGE, ["prompt", "label"]) is TaskType.IMAGE_CLASSIFICATION
    assert detect_task(Modality.IMAGE, ["prompt"]) is TaskType.IMAGE_GENERATION
    assert detect_task(Modality.AUDIO, ["text", "label"]) is TaskType.AUDIO_CLASSIFICATION
    assert detect_task(Modality.AUDIO, ["text"]) is TaskType.SPEECH_SYNTHESIS
    assert detect_task(Modality.VIDEO, ["prompt"]) is TaskType.TEXT_TO_VIDEO
    assert detect_task(Modality.GRAPH, []) is TaskType.GENERIC
