import pytest
from anodyne_evaluation import task_metrics as task_metrics_module
from anodyne_evaluation.models import DEFAULT_WEIGHTS, EvalDimension
from anodyne_evaluation.task import TaskType
from anodyne_evaluation.task_metrics import MetricSpec, catalog_for, provider_for


def test_task_quality_dimension_has_weight() -> None:
    assert EvalDimension.TASK_QUALITY.value == "task_quality"
    assert DEFAULT_WEIGHTS[EvalDimension.TASK_QUALITY] > 0


def test_unregistered_task_has_no_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    # Simulate a task type with no registered provider by removing whatever entry
    # is currently registered for it, rather than depending on some real TaskType
    # happening to be unregistered (a premise that breaks once every TaskType has
    # a provider).
    task = TaskType.TABULAR_CLASSIFICATION
    monkeypatch.delitem(task_metrics_module._REGISTRY, task, raising=False)
    assert provider_for(task) is None
    assert catalog_for(task) == []


def test_metric_spec_shape() -> None:
    spec = MetricSpec(key="accuracy", label="Accuracy", description="d", requires_llm=True)
    assert spec.key == "accuracy" and spec.requires_llm is True
