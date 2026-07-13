from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

from anodyne_dataset.models import (
    AudioSynthesisRequest,
    AudioSynthesisResult,
    DatasetSpec,
    DatasetVersion,
    ExportArtifact,
    FieldSpec,
    GenerationJob,
    PerturbationJob,
    PerturbationSpec,
    Profile,
)

if TYPE_CHECKING:
    import pyarrow  # type: ignore[import-untyped]
    from anodyne_core.ports import ObjectStore


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

    @abstractmethod
    async def get_version(self, tenant_id: UUID, version_id: UUID) -> DatasetVersion | None:
        """Look up a single version by id alone (no `dataset_id` needed).

        Additive for sub-system G (`POST /feedback`'s target is a bare
        `target_id`, with no `dataset_id` in the URL to scope
        `list_versions`). `SqlDatasetRepository` is the only real subclass of
        this ABC in the repo -- every other consumer overrides the port with a
        duck-typed fake, so this new abstract method breaks nothing else.
        """


class Generator(ABC):
    @abstractmethod
    def generate(
        self, spec: DatasetSpec, start_row: int, count: int, seed: int
    ) -> pyarrow.Table: ...


class Perturbator(ABC):
    """Applies a `PerturbationSpec` to a modality's artifact table, producing a
    corrupted copy. Deterministic + seeded exactly like `Generator.generate`:
    the same `(spec, table, modality, seed)` always yields the same output.

    Synchronous (CPU-bound); async callers run it via `asyncio.to_thread`.
    """

    @abstractmethod
    def perturb(
        self,
        spec: PerturbationSpec,
        table: pyarrow.Table,
        modality: str,
        seed: int,
    ) -> pyarrow.Table: ...


class PerturbationRepository(ABC):
    """Persists `PerturbationJob`s. Kept separate from `DatasetRepository` (like
    `ProfileRepository`) so adding it never breaks an existing repo fake."""

    @abstractmethod
    async def save_perturbation_job(self, job: PerturbationJob) -> None: ...

    @abstractmethod
    async def get_perturbation_job(
        self, tenant_id: UUID, job_id: UUID
    ) -> PerturbationJob | None: ...

    @abstractmethod
    async def list_perturbation_jobs(
        self, tenant_id: UUID, dataset_id: UUID
    ) -> list[PerturbationJob]: ...


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


class ExportRepository(ABC):
    """Persists `ExportArtifact`s. Kept separate from `DatasetRepository`, mirroring
    `ProfileRepository`, so adding it never breaks an existing `DatasetRepository`
    implementation/fake."""

    @abstractmethod
    async def add_export(self, artifact: ExportArtifact) -> None: ...

    @abstractmethod
    async def list_exports(self, tenant_id: UUID, dataset_id: UUID) -> list[ExportArtifact]: ...


class Exporter(ABC):
    """Serializes a stored `DatasetVersion` artifact to a downloadable format.

    Implemented by `anodyne_export.exporter.PyArrowExporter` (CSV/JSON/Parquet/Arrow, chunked via
    pyarrow). Takes the `ObjectStore` explicitly (like `SampleProfiler` takes raw bytes) rather than
    holding one, so callers control tenant scoping.
    """

    @abstractmethod
    async def export(
        self,
        version: DatasetVersion,
        store: ObjectStore,
        *,
        format: str | None = None,
        batch_size: int = 50_000,
    ) -> ExportArtifact: ...


class AudioProvider(ABC):
    """Port for text-to-speech / audio synthesis, implemented by both self-hosted
    OSS (e.g. XTTS/Bark, via Ray/GPU) and external-API (e.g. ElevenLabs) adapters
    in the `anodyne-audio` package."""

    @abstractmethod
    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult: ...
