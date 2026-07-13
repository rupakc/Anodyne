from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID

from pydantic import BaseModel, Field


class Modality(StrEnum):
    TABULAR = "tabular"
    TEXT = "text"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    GRAPH = "graph"


class SemanticType(StrEnum):
    INTEGER = "integer"
    FLOAT = "float"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    NAME = "name"
    EMAIL = "email"
    ADDRESS = "address"
    TEXT = "text"


class FieldSpec(BaseModel):
    name: str
    semantic_type: SemanticType
    nullable: bool = False
    constraints: dict[str, object] = Field(default_factory=dict)
    distribution: str | None = None


class DatasetSpec(BaseModel):
    id: UUID
    tenant_id: UUID
    name: str
    description: str
    modality: Modality
    source: str
    fields: list[FieldSpec]
    target_rows: int
    directives: dict[str, object] = Field(default_factory=dict)
    status: str = "draft"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class JobStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_REVIEW = "awaiting_review"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class GenerationJob(BaseModel):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    workflow_id: str | None = None


class DatasetVersion(BaseModel):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    artifact_uri: str
    format: str = "parquet"
    row_count: int = 0
    checksum: str = ""
    # Lineage: set when this version was derived from another (e.g. a
    # perturbation of `parent_version_id`). `None` for freshly generated
    # versions -- additive, so every existing caller/test is unaffected.
    parent_version_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PerturbationFamily(StrEnum):
    """The controlled-corruption families (architecture requirements 3 & 4).

    The first five are the tabular/text families; the trailing `GRAPH_*` families
    (Track GH) perturb the graph modality's node-link artifact and are dispatched
    to the graph perturbator rather than the columnar handlers.
    """

    NOISE = "noise"
    DRIFT = "drift"
    OUTLIERS = "outliers"
    BIAS = "bias"
    EDGE_CASE = "edge_case"
    # Graph modality (Track GH): structural + semantic corruption of a graph.
    GRAPH_REWIRE = "graph_rewire"
    GRAPH_DROPOUT = "graph_dropout"
    GRAPH_ONTOLOGY_VIOLATION = "graph_ontology_violation"


class PerturbationSpec(BaseModel):
    """Typed config for one perturbation: which family, how hard, on which fields.

    `params` is a free-form dict (consistent with `FieldSpec.constraints` and
    `DatasetSpec.directives`); the `anodyne_perturbation` adapter parses it into
    typed per-family param models. `target_fields` empty means "all eligible
    fields for this family". `intensity` is a 0..1 knob interpreted per family.
    """

    family: PerturbationFamily
    intensity: float = 0.1
    target_fields: list[str] = Field(default_factory=list)
    params: dict[str, object] = Field(default_factory=dict)
    # Persisted so a stored job is replayable: the same seed reproduces the exact perturbation.
    seed: int = 0


class PerturbationJob(BaseModel):
    """A durable perturbation run: parent version in, derived version out.

    Mirrors `GenerationJob` (status/progress/message/workflow_id) and adds the
    lineage (`parent_version_id`), the embedded `PerturbationSpec`, and the
    `result_version_id` stamped once the derived `DatasetVersion` is registered.
    """

    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    parent_version_id: UUID
    spec: PerturbationSpec
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    message: str = ""
    workflow_id: str | None = None
    result_version_id: UUID | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ShardArtifact(BaseModel):
    shard_index: int
    object_key: str
    row_count: int


class ColumnProfile(BaseModel):
    """Inferred schema + statistics for one column of an uploaded sample."""

    name: str
    semantic_type: SemanticType
    nullable: bool = False
    null_rate: float = 0.0
    distinct_count: int | None = None
    # Numeric stats (integer/float columns).
    min: float | None = None
    max: float | None = None
    mean: float | None = None
    std: float | None = None
    # Categorical stats: value -> relative frequency (top-K).
    categories: dict[str, float] | None = None


class Profile(BaseModel):
    """Schema + per-column distributions + correlations inferred from an uploaded sample."""

    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    row_count: int
    columns: list[ColumnProfile]
    correlations: dict[str, dict[str, float]] = Field(default_factory=dict)
    sample_uri: str
    sample_filename: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class ExportArtifact(BaseModel):
    """A `DatasetVersion` transcoded to a downloadable format (CSV/JSON/Parquet/Arrow)."""

    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    version_id: UUID
    format: str
    row_count: int
    object_key: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class AudioSynthesisRequest(BaseModel):
    """A single text-to-speech synthesis request, passed to an `AudioProvider`."""

    text: str
    voice: str | None = None
    language: str | None = None


class AudioSynthesisResult(BaseModel):
    """The audio produced for one `AudioSynthesisRequest`."""

    audio_bytes: bytes
    format: str = "wav"
    duration_seconds: float | None = None
