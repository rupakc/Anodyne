"""Evaluation domain models.

A `Judge` (an expert) produces an `ExpertScore` for one `EvalDimension`; the
`Aggregator` combines the experts' verdicts into a weighted 360-degree
`EvaluationReport`. `EvaluationRun` is the persisted lifecycle record (status,
progress, artifact locations), mirroring `GenerationJob` for the generation
side. All scores are normalized to ``0..1`` where **higher is better**, so a
single weighted mean across dimensions is meaningful.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class EvalDimension(StrEnum):
    """The mixture-of-experts dimensions (one expert judge each).

    The first block is the tabular/text mixture; the ``GRAPH_*`` block is the
    graph modality's mixture (sub-system GD). A single run only ever produces
    scores for one modality's dimensions, so the aggregator's renormalization
    keeps the 360-degree score well-defined either way.
    """

    FIDELITY = "fidelity"
    DIVERSITY = "diversity"
    PRIVACY = "privacy"
    UTILITY = "utility"
    BIAS = "bias"
    QUALITATIVE = "qualitative"
    # Graph modality (GD).
    GRAPH_STRUCTURE = "graph_structure"
    GRAPH_ONTOLOGY = "graph_ontology"
    GRAPH_SEMANTIC = "graph_semantic"
    GRAPH_CONNECTIVITY = "graph_connectivity"
    GRAPH_UTILITY = "graph_utility"
    GRAPH_PRIVACY = "graph_privacy"
    # Task-quality expert (sub-system F): per-`TaskType` standard metrics, scored
    # by whichever `TaskMetricProvider` is registered for the run's task (see
    # `task_metrics.py`). Present in every modality's weight group.
    TASK_QUALITY = "task_quality"


# Default 360-degree weights, grouped by modality. Each modality's group sums to
# 1.0 on its own; the aggregator renormalizes over whichever dimensions actually
# produced a score, so mixing never happens in practice (a run is single-modality).
# Overridable per run via `EvaluationConfig.weights`.
TABULAR_WEIGHTS: dict[str, float] = {
    EvalDimension.FIDELITY: 0.22,
    EvalDimension.PRIVACY: 0.17,
    EvalDimension.UTILITY: 0.17,
    EvalDimension.DIVERSITY: 0.12,
    EvalDimension.QUALITATIVE: 0.08,
    EvalDimension.BIAS: 0.09,
    EvalDimension.TASK_QUALITY: 0.15,
}
GRAPH_WEIGHTS: dict[str, float] = {
    EvalDimension.GRAPH_STRUCTURE: 0.22,
    EvalDimension.GRAPH_ONTOLOGY: 0.18,
    EvalDimension.GRAPH_PRIVACY: 0.13,
    EvalDimension.GRAPH_CONNECTIVITY: 0.13,
    EvalDimension.GRAPH_UTILITY: 0.13,
    EvalDimension.GRAPH_SEMANTIC: 0.08,
    EvalDimension.TASK_QUALITY: 0.13,
}
# Media modalities (image/audio/video) have no bespoke statistical experts yet;
# the mixture is task-quality (standard per-task metrics) plus the qualitative
# LLM judge.
MEDIA_WEIGHTS: dict[str, float] = {
    EvalDimension.TASK_QUALITY: 0.7,
    EvalDimension.QUALITATIVE: 0.3,
}
DEFAULT_WEIGHTS: dict[str, float] = {**TABULAR_WEIGHTS, **GRAPH_WEIGHTS, **MEDIA_WEIGHTS}


class ExpertScore(BaseModel):
    """One expert judge's verdict for a single dimension."""

    dimension: EvalDimension
    score: float  # normalized 0..1, higher is better
    rationale: str
    metrics: dict[str, float] = Field(default_factory=dict)  # raw underlying numbers
    recommendations: list[str] = Field(default_factory=list)


class EvaluationConfig(BaseModel):
    """Per-run knobs. Carried verbatim onto the workflow input + persisted on the run."""

    weights: dict[str, float] = Field(default_factory=dict)  # dimension -> weight overrides
    sensitive_field: str | None = None  # enables BiasJudge
    target_field: str | None = None  # enables UtilityJudge (TSTR) + bias outcome disparity
    text_column: str | None = None  # column sampled for the qualitative LLM judge (tabular/text)
    model_config_id: UUID | None = None  # which registered LLM drives the qualitative judge
    sample_rows: int = 20  # rows sampled for the qualitative rubric prompt
    seed: int = 0


class EvaluationReport(BaseModel):
    """The weighted 360-degree report: per-expert breakdown + overall + recommendations."""

    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    dataset_version_id: UUID
    reference_version_id: UUID | None = None
    overall_score: float
    expert_scores: list[ExpertScore]
    weights: dict[str, float]  # the (renormalized) weights actually applied
    recommendations: list[str] = Field(default_factory=list)
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class EvaluationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class EvaluationRun(BaseModel):
    """Persisted lifecycle record for one evaluation (mirrors `GenerationJob`)."""

    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    dataset_version_id: UUID
    reference_version_id: UUID | None = None
    status: EvaluationStatus = EvaluationStatus.PENDING
    progress: float = 0.0
    message: str = ""
    workflow_id: str | None = None
    report_uri: str | None = None  # object-store key of the JSON report
    report_html_uri: str | None = None  # object-store key of the HTML report
    overall_score: float | None = None
    config: dict[str, object] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
