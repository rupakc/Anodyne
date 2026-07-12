"""SQL-backed `DatasetRepository`: dataset specs, generation jobs, and versions.

Mirrors `anodyne_llm.registry.SqlModelRegistry`: every method runs inside a
`tenant_session` (which sets the RLS `app.tenant_id` GUC via `SET LOCAL`), and
reads add an explicit `tenant_id` filter as defense-in-depth on top of RLS.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from anodyne_dataset.models import DatasetSpec, DatasetVersion, FieldSpec, GenerationJob
from anodyne_dataset.ports import DatasetRepository
from sqlalchemy import select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from anodyne_storage.db import dataset_versions, datasets, generation_jobs, tenant_session


def _spec_from_row(m: Any) -> DatasetSpec:
    return DatasetSpec(
        id=m["id"],
        tenant_id=m["tenant_id"],
        name=m["name"],
        description=m["description"],
        modality=m["modality"],
        source=m["source"],
        fields=[FieldSpec.model_validate(f) for f in m["field_specs"]],
        target_rows=m["target_rows"],
        directives=m["directives"],
        status=m["status"],
        created_at=m["created_at"],
    )


def _job_from_row(m: Any) -> GenerationJob:
    return GenerationJob(
        id=m["id"],
        tenant_id=m["tenant_id"],
        dataset_id=m["dataset_id"],
        status=m["status"],
        progress=m["progress"],
        message=m["message"],
        workflow_id=m["workflow_id"],
    )


def _version_from_row(m: Any) -> DatasetVersion:
    return DatasetVersion(
        id=m["id"],
        tenant_id=m["tenant_id"],
        dataset_id=m["dataset_id"],
        artifact_uri=m["artifact_uri"],
        format=m["format"],
        row_count=m["row_count"],
        checksum=m["checksum"],
        created_at=m["created_at"],
    )


class SqlDatasetRepository(DatasetRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def create_spec(self, spec: DatasetSpec) -> None:
        async with tenant_session(self._engine, spec.tenant_id) as s:
            await s.execute(
                datasets.insert().values(
                    id=spec.id,
                    tenant_id=spec.tenant_id,
                    name=spec.name,
                    description=spec.description,
                    modality=str(spec.modality),
                    source=spec.source,
                    field_specs=[f.model_dump(mode="json") for f in spec.fields],
                    target_rows=spec.target_rows,
                    directives=spec.directives,
                    status=spec.status,
                    created_at=spec.created_at,
                )
            )
            await s.commit()

    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        async with tenant_session(self._engine, tenant_id) as s:
            row = (
                (
                    await s.execute(
                        select(datasets).where(
                            datasets.c.id == dataset_id,
                            datasets.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .first()
            )
            return _spec_from_row(row) if row else None

    async def list_specs(self, tenant_id: UUID) -> list[DatasetSpec]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                (await s.execute(select(datasets).where(datasets.c.tenant_id == tenant_id)))
                .mappings()
                .all()
            )
            return [_spec_from_row(r) for r in rows]

    async def update_spec(self, spec: DatasetSpec) -> None:
        async with tenant_session(self._engine, spec.tenant_id) as s:
            await s.execute(
                update(datasets)
                .where(
                    datasets.c.id == spec.id,
                    datasets.c.tenant_id == spec.tenant_id,
                )
                .values(
                    name=spec.name,
                    description=spec.description,
                    modality=str(spec.modality),
                    source=spec.source,
                    field_specs=[f.model_dump(mode="json") for f in spec.fields],
                    target_rows=spec.target_rows,
                    directives=spec.directives,
                    status=spec.status,
                )
            )
            await s.commit()

    async def save_job(self, job: GenerationJob) -> None:
        async with tenant_session(self._engine, job.tenant_id) as s:
            values = {
                "id": job.id,
                "tenant_id": job.tenant_id,
                "dataset_id": job.dataset_id,
                "status": str(job.status),
                "progress": job.progress,
                "message": job.message,
                "workflow_id": job.workflow_id,
            }
            stmt = pg_insert(generation_jobs).values(**values)
            stmt = stmt.on_conflict_do_update(
                index_elements=[generation_jobs.c.id],
                set_={k: v for k, v in values.items() if k != "id"},
            )
            await s.execute(stmt)
            await s.commit()

    async def get_job(self, tenant_id: UUID, job_id: UUID) -> GenerationJob | None:
        async with tenant_session(self._engine, tenant_id) as s:
            row = (
                (
                    await s.execute(
                        select(generation_jobs).where(
                            generation_jobs.c.id == job_id,
                            generation_jobs.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .first()
            )
            return _job_from_row(row) if row else None

    async def add_version(self, version: DatasetVersion) -> None:
        async with tenant_session(self._engine, version.tenant_id) as s:
            await s.execute(
                dataset_versions.insert().values(
                    id=version.id,
                    tenant_id=version.tenant_id,
                    dataset_id=version.dataset_id,
                    artifact_uri=version.artifact_uri,
                    format=version.format,
                    row_count=version.row_count,
                    checksum=version.checksum,
                    created_at=version.created_at,
                )
            )
            await s.commit()

    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                (
                    await s.execute(
                        select(dataset_versions).where(
                            dataset_versions.c.dataset_id == dataset_id,
                            dataset_versions.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            return [_version_from_row(r) for r in rows]
