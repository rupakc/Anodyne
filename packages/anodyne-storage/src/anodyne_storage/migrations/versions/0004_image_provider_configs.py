"""image_provider_configs: per-tenant image-provider registrations + RLS policy

Revision ID: 0003
Revises: 0002
Create Date: 2026-07-12

Mirrors `model_configs` exactly (same columns) but is a **separate** table:
kept apart from the LLM model registry so an image-provider registration can
never be accidentally picked up as "the tenant's LLM" by
`api_gateway.deps.get_schema_proposer`'s `configs[0]` -- see
`docs/superpowers/specs/2026-07-12-generation-c3-design.md`.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from anodyne_storage.db import metadata

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_TABLES: dict[str, str] = {
    "image_provider_configs": "tenant_id",
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
