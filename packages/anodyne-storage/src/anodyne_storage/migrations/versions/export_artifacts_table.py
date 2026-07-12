"""export_artifacts: per-dataset-version transcoded export records + RLS

Revision ID: export_artifacts
Revises: 0006
Create Date: 2026-07-12

Sub-system E (Export & Storage). Records every `ExportArtifact` produced by
`anodyne_export.PyArrowExporter().export(...)` (CSV/JSON/Parquet/Arrow), mirroring
`dataset_versions`'s shape. Kept as its own table (rather than growing
`dataset_versions`) since an export is a *derived* artifact of a version, not a
version itself, and a version can have zero-to-many exports in different formats.

Named descriptively per the sub-system E task brief rather than the numeric
`0007_*` convention the other migrations use, to avoid colliding with a numeric
revision another in-flight branch might also claim.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from anodyne_storage.db import metadata

# revision identifiers, used by Alembic.
revision: str = "export_artifacts"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_TABLES: dict[str, str] = {
    "export_artifacts": "tenant_id",
}


def upgrade() -> None:
    bind = op.get_bind()
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
