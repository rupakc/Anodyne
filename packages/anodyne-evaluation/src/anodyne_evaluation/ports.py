"""Evaluation ports (the hexagonal boundary for sub-system F).

Kept adapter-free and light: pandas is imported only under ``TYPE_CHECKING`` so
importing this module never pulls in the heavy numeric stack. `anodyne-core`
imports nothing from here -- the `Judge`/`Aggregator`/`EvaluationRepository`
ports live in this bounded-context package exactly like `anodyne_video.ports`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from uuid import UUID

from anodyne_dataset.models import Modality

from anodyne_evaluation.models import EvalDimension, EvaluationReport, EvaluationRun, ExpertScore

if TYPE_CHECKING:
    import pandas as pd  # type: ignore[import-untyped]


class JudgeNotApplicable(Exception):
    """Raised by a `Judge` when its preconditions are unmet for this run.

    The aggregator excludes the dimension and renormalizes the remaining
    weights, so the 360-degree score always reflects only the dimensions that
    could actually be measured (e.g. fidelity needs a reference dataset; bias
    needs a sensitive field).
    """


@dataclass(frozen=True)
class EvaluationContext:
    """Immutable, in-memory inputs handed to every `Judge`.

    Carries DataFrames (not pydantic) so it can hold the loaded artifacts
    directly; the qualitative judge receives its `LLMProvider`/`ModelConfig`
    through its own constructor rather than here, keeping this data-only.
    """

    subject: pd.DataFrame  # the dataset version under evaluation
    reference: pd.DataFrame | None = None  # the "real"/reference version, if provided
    modality: Modality = Modality.TABULAR
    sensitive_field: str | None = None
    target_field: str | None = None
    text_column: str | None = None
    sample_rows: int = 20
    seed: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


class Judge(ABC):
    """A single expert in the mixture. Async so the LLM-backed expert fits the
    same port as the CPU-bound statistical ones."""

    dimension: EvalDimension

    @abstractmethod
    async def evaluate(self, ctx: EvaluationContext) -> ExpertScore:
        """Score `ctx` on this expert's dimension, or raise `JudgeNotApplicable`."""


# A JudgeRunner runs the experts and returns the scores of those that applied
# (JudgeNotApplicable ones are dropped). `sequential_runner` is the offline
# default; `RayJudgeRunner` (anodyne-compute) is the parallel production path.
JudgeRunner = Callable[[Sequence[Judge], EvaluationContext], Awaitable[list[ExpertScore]]]


class Aggregator(ABC):
    @abstractmethod
    def aggregate(
        self,
        scores: list[ExpertScore],
        *,
        tenant_id: UUID,
        dataset_id: UUID,
        dataset_version_id: UUID,
        reference_version_id: UUID | None,
        weights: dict[str, float] | None = None,
    ) -> EvaluationReport:
        """Combine expert verdicts into a weighted 360-degree report."""


class EvaluationRepository(ABC):
    """Persists evaluation runs + per-expert results. Separate from
    `DatasetRepository` so adding it never breaks an existing implementation."""

    @abstractmethod
    async def create_run(self, run: EvaluationRun) -> None: ...

    @abstractmethod
    async def save_run(self, run: EvaluationRun) -> None: ...

    @abstractmethod
    async def get_run(self, tenant_id: UUID, run_id: UUID) -> EvaluationRun | None: ...

    @abstractmethod
    async def list_runs(self, tenant_id: UUID, dataset_id: UUID) -> list[EvaluationRun]: ...

    @abstractmethod
    async def add_expert_results(
        self, tenant_id: UUID, run_id: UUID, scores: list[ExpertScore]
    ) -> None: ...

    @abstractmethod
    async def get_expert_results(self, tenant_id: UUID, run_id: UUID) -> list[ExpertScore]: ...
