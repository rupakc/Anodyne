"""evaluation_runs + evaluation_expert_results: MoE LLM-as-a-Judge results + RLS

Revision ID: 0007
Revises: 0006
Create Date: 2026-07-12

Sub-system F (Evaluation Engine). Adds the two evaluation tables the same way
`0006` added `audio_provider_configs`: create from the shared `metadata`, then
enable+force RLS and install the per-tenant isolation policy. Both tables carry
`tenant_id` and are keyed by it in the policy.
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op
from anodyne_storage.db import metadata

# revision identifiers, used by Alembic.
revision: str = "0007"
down_revision: str | None = "perturbation_jobs"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_TENANT_TABLES: dict[str, str] = {
    "evaluation_runs": "tenant_id",
    "evaluation_expert_results": "tenant_id",
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
