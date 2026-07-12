"""datasets: datasets/generation_jobs/dataset_versions tables + RLS policies

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-12

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from anodyne_storage.db import metadata

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tenant-scoped tables added by this migration and the column each RLS policy
# is keyed on. Kept in sync with `anodyne_storage.db._TENANT_TABLES` (duplicated
# here rather than imported since `apply_rls` is async and this migration runs
# synchronously).
_TENANT_TABLES: dict[str, str] = {
    "datasets": "tenant_id",
    "generation_jobs": "tenant_id",
    "dataset_versions": "tenant_id",
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
