"""annotations + feedback + review_tasks: human-in-the-loop & annotation + RLS

Revision ID: 0008
Revises: 0007
Create Date: 2026-07-12

Sub-system G (Human-in-the-Loop & Annotation). Adds the three tables the same
way `0007` added the evaluation tables: create from the shared `metadata`,
then enable+force RLS and install the per-tenant isolation policy. All three
carry `tenant_id` and are keyed by it in the policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from anodyne_storage.db import metadata

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_TABLES: dict[str, str] = {
    "annotations": "tenant_id",
    "feedback": "tenant_id",
    "review_tasks": "tenant_id",
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
