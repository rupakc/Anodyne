from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

from anodyne_dataset.models import (
    AudioSynthesisRequest,
    AudioSynthesisResult,
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    GenerationJob,
)

if TYPE_CHECKING:
    import pyarrow  # type: ignore[import-not-found, import-untyped, unused-ignore]


class DatasetRepository(ABC):
    @abstractmethod
    async def create_spec(self, spec: DatasetSpec) -> None: ...

    @abstractmethod
    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None: ...

    @abstractmethod
    async def list_specs(self, tenant_id: UUID) -> list[DatasetSpec]: ...

    @abstractmethod
    async def update_spec(self, spec: DatasetSpec) -> None: ...

    @abstractmethod
    async def save_job(self, job: GenerationJob) -> None: ...

    @abstractmethod
    async def get_job(self, tenant_id: UUID, job_id: UUID) -> GenerationJob | None: ...

    @abstractmethod
    async def add_version(self, version: DatasetVersion) -> None: ...

    @abstractmethod
    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]: ...


class Generator(ABC):
    @abstractmethod
    def generate(
        self, spec: DatasetSpec, start_row: int, count: int, seed: int
    ) -> pyarrow.Table: ...


class SchemaProposer(ABC):
    @abstractmethod
    async def propose(self, description: str) -> list[FieldSpec]: ...


class AudioProvider(ABC):
    """Port for text-to-speech / audio synthesis, implemented by both self-hosted
    OSS (e.g. XTTS/Bark, via Ray/GPU) and external-API (e.g. ElevenLabs) adapters
    in the `anodyne-audio` package."""

    @abstractmethod
    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult: ...
