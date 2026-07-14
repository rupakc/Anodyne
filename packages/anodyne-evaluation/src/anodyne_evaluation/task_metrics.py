"""Per-task metric catalog + provider registry (sub-system F, task-quality mixture).

A `TaskMetricProvider` knows the standard metrics for one `TaskType` (e.g. QA,
summarization, tabular classification) and how to score a run against them,
producing a single `ExpertScore` for the `EvalDimension.TASK_QUALITY` expert.
Providers register themselves at import time via `register_provider`; the
registry is populated by task-specific provider modules added in later tasks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from anodyne_core.models import ModelConfig
from anodyne_core.ports import LLMProvider
from pydantic import BaseModel

from anodyne_evaluation.models import ExpertScore
from anodyne_evaluation.task import TaskType

if TYPE_CHECKING:
    from anodyne_evaluation.ports import EvaluationContext


class MetricSpec(BaseModel):
    """Describes one standard metric offered for a task type."""

    key: str
    label: str
    description: str
    requires_llm: bool = False


@runtime_checkable
class TaskMetricProvider(Protocol):
    """A task-specific source of standard metrics + the TASK_QUALITY expert score."""

    task_type: TaskType

    def metric_catalog(self) -> list[MetricSpec]: ...

    async def score(
        self,
        ctx: EvaluationContext,
        provider: LLMProvider,
        model_config: ModelConfig,
        *,
        selected: frozenset[str],
    ) -> ExpertScore: ...


_REGISTRY: dict[TaskType, TaskMetricProvider] = {}


def register_provider(p: TaskMetricProvider) -> None:
    """Register (or replace) the provider for its `task_type`."""
    _REGISTRY[p.task_type] = p


def provider_for(task: TaskType) -> TaskMetricProvider | None:
    """Look up the registered provider for `task`, if any."""
    return _REGISTRY.get(task)


def catalog_for(task: TaskType) -> list[MetricSpec]:
    """The metric catalog for `task`, or `[]` if no provider is registered."""
    p = _REGISTRY.get(task)
    return p.metric_catalog() if p else []
