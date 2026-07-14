import pandas as pd
from anodyne_evaluation.models import EvaluationConfig
from anodyne_evaluation.ports import EvaluationContext
from anodyne_evaluation.task import TaskType


def test_context_carries_task_fields() -> None:
    ctx = EvaluationContext(
        subject=pd.DataFrame(), task_type=TaskType.QA, selected_metrics=frozenset({"groundedness"})
    )
    assert ctx.task_type is TaskType.QA
    assert ctx.selected_metrics == frozenset({"groundedness"})


def test_config_defaults() -> None:
    cfg = EvaluationConfig()
    assert cfg.task_type is None and cfg.selected_metrics is None
