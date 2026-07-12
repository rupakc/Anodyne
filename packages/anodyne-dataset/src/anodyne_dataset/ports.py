from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    GenerationJob,
    Profile,
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


class SampleProfiler(ABC):
    """Infers a `Profile` (schema + distributions + correlations) from an uploaded sample.

    Synchronous like `Generator.generate` (CPU-bound); async callers should run it via
    `asyncio.to_thread`.
    """

    @abstractmethod
    def profile(
        self, tenant_id: UUID, dataset_id: UUID, sample_uri: str, data: bytes, filename: str
    ) -> Profile: ...


class ProfileRepository(ABC):
    """Persists `Profile`s. Kept separate from `DatasetRepository` so adding it never breaks an
    existing `DatasetRepository` implementation/fake."""

    @abstractmethod
    async def save_profile(self, profile: Profile) -> None: ...

    @abstractmethod
    async def get_profile(self, tenant_id: UUID, dataset_id: UUID) -> Profile | None: ...
