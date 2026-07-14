from anodyne_evaluation.models import DEFAULT_WEIGHTS, EvalDimension
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, catalog_for, provider_for


def test_task_quality_dimension_has_weight() -> None:
    assert EvalDimension.TASK_QUALITY.value == "task_quality"
    assert DEFAULT_WEIGHTS[EvalDimension.TASK_QUALITY] > 0


def test_unregistered_task_has_no_provider() -> None:
    # tabular_classification's provider is a later task; text_classification/qa/
    # summarization/chat/generic are registered by this point.
    assert provider_for(TaskType.TABULAR_CLASSIFICATION) is None
    assert catalog_for(TaskType.TABULAR_CLASSIFICATION) == []


def test_metric_spec_shape() -> None:
    spec = MetricSpec(key="accuracy", label="Accuracy", description="d", requires_llm=True)
    assert spec.key == "accuracy" and spec.requires_llm is True
