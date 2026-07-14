from anodyne_evaluation.models import DEFAULT_WEIGHTS, EvalDimension
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, catalog_for, provider_for


def test_task_quality_dimension_has_weight() -> None:
    assert EvalDimension.TASK_QUALITY.value == "task_quality"
    assert DEFAULT_WEIGHTS[EvalDimension.TASK_QUALITY] > 0


def test_unregistered_task_has_no_provider() -> None:
    # No providers are registered yet -- the generic provider lands in Task 4.
    assert provider_for(TaskType.QA) is None
    assert catalog_for(TaskType.QA) == []


def test_metric_spec_shape() -> None:
    spec = MetricSpec(key="accuracy", label="Accuracy", description="d", requires_llm=True)
    assert spec.key == "accuracy" and spec.requires_llm is True
