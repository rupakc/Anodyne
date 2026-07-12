from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, Field


class AudioManifestItem(BaseModel):
    """One entry in an audio dataset's JSON manifest: metadata for a generated clip."""

    index: int
    object_key: str
    text: str
    label: str | None = None
    voice: str | None = None
    format: str = "wav"
    duration_seconds: float | None = None


class AudioManifest(BaseModel):
    """The assembled manifest for a completed audio generation job."""

    dataset_id: UUID
    job_id: UUID
    items: list[AudioManifestItem] = Field(default_factory=list)
