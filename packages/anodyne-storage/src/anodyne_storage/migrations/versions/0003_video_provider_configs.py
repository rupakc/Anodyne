"""video_provider_configs: per-tenant video-generation provider configs + RLS

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-12

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from anodyne_storage.db import metadata

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Kept in sync with `anodyne_storage.db._TENANT_TABLES` (duplicated here since
# `apply_rls` is async and this migration runs synchronously -- see
# `0002_datasets.py` for the same pattern).
_TENANT_TABLES: dict[str, str] = {
    "video_provider_configs": "tenant_id",
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
