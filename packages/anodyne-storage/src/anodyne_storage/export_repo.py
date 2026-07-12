"""SQL-backed `ExportRepository`: persists `ExportArtifact`s produced by `anodyne_export`.

Mirrors `SqlDatasetRepository.add_version`/`list_versions` exactly: every method runs inside a
`tenant_session` (RLS `app.tenant_id` GUC via `SET LOCAL`), and reads add an explicit `tenant_id`
filter as defense-in-depth on top of RLS.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from anodyne_dataset.models import ExportArtifact
from anodyne_dataset.ports import ExportRepository
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from anodyne_storage.db import export_artifacts, tenant_session


def _artifact_from_row(m: Any) -> ExportArtifact:
    return ExportArtifact(
        id=m["id"],
        tenant_id=m["tenant_id"],
        dataset_id=m["dataset_id"],
        version_id=m["version_id"],
        format=m["format"],
        row_count=m["row_count"],
        object_key=m["object_key"],
        created_at=m["created_at"],
    )


class SqlExportRepository(ExportRepository):
    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine

    async def add_export(self, artifact: ExportArtifact) -> None:
        async with tenant_session(self._engine, artifact.tenant_id) as s:
            await s.execute(
                export_artifacts.insert().values(
                    id=artifact.id,
                    tenant_id=artifact.tenant_id,
                    dataset_id=artifact.dataset_id,
                    version_id=artifact.version_id,
                    format=artifact.format,
                    row_count=artifact.row_count,
                    object_key=artifact.object_key,
                    created_at=artifact.created_at,
                )
            )
            await s.commit()

    async def list_exports(self, tenant_id: UUID, dataset_id: UUID) -> list[ExportArtifact]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                (
                    await s.execute(
                        select(export_artifacts).where(
                            export_artifacts.c.dataset_id == dataset_id,
                            export_artifacts.c.tenant_id == tenant_id,
                        )
                    )
                )
                .mappings()
                .all()
            )
            return [_artifact_from_row(r) for r in rows]
