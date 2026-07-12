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

from sqlalchemy import Column, DateTime, Float, Integer, MetaData, String, Table, Text, func, text
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

video_provider_configs = Table(
    "video_provider_configs",
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

datasets = Table(
    "datasets",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("name", String, nullable=False),
    Column("description", Text, nullable=False),
    Column("modality", String, nullable=False),
    Column("source", String, nullable=False),
    Column("field_specs", JSONB, nullable=False, server_default="[]"),
    Column("target_rows", Integer, nullable=False),
    Column("directives", JSONB, nullable=False, server_default="{}"),
    Column("status", String, nullable=False, server_default="draft"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

generation_jobs = Table(
    "generation_jobs",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("dataset_id", PgUUID(as_uuid=True), nullable=False),
    Column("status", String, nullable=False, server_default="pending"),
    Column("progress", Float, nullable=False, server_default="0.0"),
    Column("message", Text, nullable=False, server_default=""),
    Column("workflow_id", String, nullable=True),
)

image_provider_configs = Table(
    "image_provider_configs",
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

dataset_versions = Table(
    "dataset_versions",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("dataset_id", PgUUID(as_uuid=True), nullable=False),
    Column("artifact_uri", String, nullable=False),
    Column("format", String, nullable=False, server_default="parquet"),
    Column("row_count", Integer, nullable=False, server_default="0"),
    Column("checksum", String, nullable=False, server_default=""),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# One profile per dataset: `dataset_id` is the primary key, so re-uploading a sample
# replaces the previous profile via upsert rather than accumulating rows.
dataset_profiles = Table(
    "dataset_profiles",
    metadata,
    Column("dataset_id", PgUUID(as_uuid=True), primary_key=True),
    Column("id", PgUUID(as_uuid=True), nullable=False),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("row_count", Integer, nullable=False),
    Column("columns", JSONB, nullable=False),
    Column("correlations", JSONB, nullable=False, server_default="{}"),
    Column("sample_uri", String, nullable=False),
    Column("sample_filename", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Tenant-scoped tables get an RLS policy keyed on the per-transaction
# app.tenant_id GUC. `tenants` is keyed by its own `id`; everything else by
# its `tenant_id` foreign key.
_TENANT_TABLES: dict[str, str] = {
    "tenants": "id",
    "users": "tenant_id",
    "model_configs": "tenant_id",
    "datasets": "tenant_id",
    "generation_jobs": "tenant_id",
    "dataset_versions": "tenant_id",
    "dataset_profiles": "tenant_id",
    "image_provider_configs": "tenant_id",
    "video_provider_configs": "tenant_id",
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
