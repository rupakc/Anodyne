"""perturbation_jobs table + dataset_versions.parent_version_id lineage + RLS

Revision ID: perturbation_jobs
Revises: 0006
Create Date: 2026-07-12

Sub-system D (Perturbation Module). Adds:
  * a nullable `dataset_versions.parent_version_id` column recording that a
    version was derived from another (a perturbation of its parent), and
  * a `perturbation_jobs` table (mirrors `generation_jobs` + the perturbation
    config family/params/intensity/target_fields + lineage + result), with the
    same per-tenant RLS policy as every other tenant table.

Deliberately given a descriptive filename/revision id rather than `0007_*`: the
generation modalities were built in parallel branches all claiming `0007`, so a
non-numeric id keeps this migration from colliding with them at integration.
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from anodyne_storage.db import metadata
from sqlalchemy.dialects.postgresql import UUID as PgUUID

# revision identifiers, used by Alembic.
revision: str = "perturbation_jobs"
down_revision: str | None = "export_artifacts"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_TABLES: dict[str, str] = {
    "perturbation_jobs": "tenant_id",
}


def upgrade() -> None:
    bind = op.get_bind()
    op.add_column(
        "dataset_versions",
        sa.Column("parent_version_id", PgUUID(as_uuid=True), nullable=True),
    )
    tables = [metadata.tables[name] for name in _TENANT_TABLES]
    metadata.create_all(bind, tables=tables)

    for tbl, col in _TENANT_TABLES.items():
        op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {tbl} USING "
            f"({col} = current_setting('app.tenant_id', true)::uuid)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    tables = [metadata.tables[name] for name in _TENANT_TABLES]
    metadata.drop_all(bind, tables=tables)
    op.drop_column("dataset_versions", "parent_version_id")
