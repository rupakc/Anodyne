"""initial: tenants/users/model_configs tables + RLS policies

Revision ID: 0001
Revises:
Create Date: 2026-07-12

"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from anodyne_storage.db import metadata

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Tenant-scoped tables and the column each RLS policy is keyed on. Kept in
# sync with `anodyne_storage.db._TENANT_TABLES` (duplicated here rather than
# imported since `apply_rls` is async and this migration runs synchronously).
_TENANT_TABLES: dict[str, str] = {
    "tenants": "id",
    "users": "tenant_id",
    "model_configs": "tenant_id",
}


def upgrade() -> None:
    bind = op.get_bind()
    metadata.create_all(bind)

    for tbl, col in _TENANT_TABLES.items():
        op.execute(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY")
        op.execute(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY")
        op.execute(
            f"CREATE POLICY tenant_isolation ON {tbl} USING "
            f"({col} = current_setting('app.tenant_id', true)::uuid)"
        )


def downgrade() -> None:
    bind = op.get_bind()
    metadata.drop_all(bind)
