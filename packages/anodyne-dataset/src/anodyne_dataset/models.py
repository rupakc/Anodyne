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
