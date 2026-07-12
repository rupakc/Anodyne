"""Async SQLAlchemy engine, ORM tables, and RLS-enforced tenant sessions.

Postgres row-level security (RLS) is the tenant-isolation boundary at the data
layer: every tenant-scoped table carries a policy keyed on the per-transaction
``app.tenant_id`` GUC, and `tenant_session` sets that GUC with ``SET LOCAL`` so
it never leaks across transactions or connections in the pool.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from uuid import UUID

from sqlalchemy import Column, MetaData, String, Table, Text, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PgUUID
from sqlalchemy.ext.asyncio import (
    AsyncConnection,
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

metadata = MetaData()

tenants = Table(
    "tenants",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("name", String, nullable=False),
    Column("org_ref", String, nullable=False),
    Column("status", String, nullable=False, server_default="active"),
)

users = Table(
    "users",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("subject", String, nullable=False),
    Column("email", String, nullable=False),
)

model_configs = Table(
    "model_configs",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("name", String, nullable=False),
    Column("provider", String, nullable=False),
    Column("model", String, nullable=False),
    Column("params", JSONB, nullable=False, server_default="{}"),
    Column("secret_ref", Text, nullable=True),
    Column("api_base", String, nullable=True),
    Column("enabled", String, nullable=False, server_default="true"),
)

# Tenant-scoped tables get an RLS policy keyed on the per-transaction
# app.tenant_id GUC. `tenants` is keyed by its own `id`; everything else by
# its `tenant_id` foreign key.
_TENANT_TABLES: dict[str, str] = {
    "tenants": "id",
    "users": "tenant_id",
    "model_configs": "tenant_id",
}


async def apply_rls(conn: AsyncConnection) -> None:
    """Enable + force RLS and create the tenant-isolation policy on each tenant table.

    Called from the Alembic migration (synchronously, via `op.execute`) and
    from tests that stand up an ad-hoc schema against a live Postgres.
    """
    for tbl, col in _TENANT_TABLES.items():
        await conn.execute(text(f"ALTER TABLE {tbl} ENABLE ROW LEVEL SECURITY"))
        await conn.execute(text(f"ALTER TABLE {tbl} FORCE ROW LEVEL SECURITY"))
        await conn.execute(
            text(
                f"CREATE POLICY tenant_isolation ON {tbl} USING "
                f"({col} = current_setting('app.tenant_id', true)::uuid)"
            )
        )


def make_engine(dsn: str) -> AsyncEngine:
    return create_async_engine(dsn, pool_pre_ping=True)


@asynccontextmanager
async def tenant_session(engine: AsyncEngine, tenant_id: UUID) -> AsyncIterator[AsyncSession]:
    """Yield an `AsyncSession` scoped to `tenant_id` for the lifetime of one transaction.

    Sets `app.tenant_id` with `SET LOCAL`, so the setting is transaction-local
    and never bleeds into another tenant's request on a pooled connection.
    """
    maker = async_sessionmaker(engine, expire_on_commit=False)
    async with maker() as session:
        # Postgres rejects bind parameters in `SET`/`SET LOCAL`; `set_config(_, _, true)`
        # is the parameterizable, transaction-local equivalent.
        await session.execute(
            text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(tenant_id)}
        )
        yield session
