from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from pydantic import BaseModel, Field


class VideoProviderConfig(BaseModel):
    """A tenant's registered video-generation provider.

    Mirrors `anodyne_core.models.ModelConfig`.
    """

    id: UUID
    tenant_id: UUID
    name: str
    provider: str  # e.g. "replicate", "self-hosted"
    model: str  # e.g. "svd-xt", "stable-video-diffusion"
    params: dict[str, object] = Field(default_factory=dict)
    secret_ref: str | None = None  # encrypted-secret handle; None for keyless self-hosted
    api_base: str | None = None
    enabled: bool = True


class VideoGenerationRequest(BaseModel):
    """One text-to-video generation request."""

    prompt: str
    duration_seconds: float = 4.0
    width: int = 576
    height: int = 320
    fps: int = 8
    seed: int = 0
    params: dict[str, object] = Field(default_factory=dict)


class VideoAsset(BaseModel):
    """One generated clip, as returned by a `VideoProvider`."""

    content: bytes
    content_type: str = "video/mp4"
    duration_seconds: float
    width: int
    height: int
    fps: int
    seed: int
    provider: str
    model: str


class VideoManifestItem(BaseModel):
    """One row of a video dataset's manifest -- metadata only, no raw bytes."""

    index: int
    object_key: str
    prompt: str
    duration_seconds: float
    width: int
    height: int
    fps: int
    seed: int
    provider: str
    model: str
    content_type: str = "video/mp4"
    byte_size: int


class VideoManifest(BaseModel):
    """The JSON manifest describing every clip generated for one job."""

    tenant_id: UUID
    dataset_id: UUID
    job_id: UUID
    items: list[VideoManifestItem] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
