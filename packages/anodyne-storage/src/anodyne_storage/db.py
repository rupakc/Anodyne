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

audio_provider_configs = Table(
    "audio_provider_configs",
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
    # Lineage: the version this one was derived from (e.g. a perturbation).
    # NULL for freshly generated versions.
    Column("parent_version_id", PgUUID(as_uuid=True), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Perturbation jobs: a durable run that reads `parent_version_id` and writes a
# new derived `DatasetVersion` (`result_version_id`). Mirrors `generation_jobs`
# plus the perturbation config (family/params/intensity/target_fields) + lineage.
perturbation_jobs = Table(
    "perturbation_jobs",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("dataset_id", PgUUID(as_uuid=True), nullable=False),
    Column("parent_version_id", PgUUID(as_uuid=True), nullable=False),
    Column("family", String, nullable=False),
    Column("params", JSONB, nullable=False, server_default="{}"),
    Column("intensity", Float, nullable=False, server_default="0.1"),
    Column("target_fields", JSONB, nullable=False, server_default="[]"),
    Column("seed", Integer, nullable=False, server_default="0"),
    Column("status", String, nullable=False, server_default="pending"),
    Column("progress", Float, nullable=False, server_default="0.0"),
    Column("message", Text, nullable=False, server_default=""),
    Column("workflow_id", String, nullable=True),
    Column("result_version_id", PgUUID(as_uuid=True), nullable=True),
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

export_artifacts = Table(
    "export_artifacts",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("dataset_id", PgUUID(as_uuid=True), nullable=False),
    Column("version_id", PgUUID(as_uuid=True), nullable=False),
    Column("format", String, nullable=False),
    Column("row_count", Integer, nullable=False, server_default="0"),
    Column("object_key", String, nullable=False),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Evaluation runs (sub-system F): one row per LLM-as-a-Judge MoE evaluation of a
# dataset version, mirroring `generation_jobs`' lifecycle shape (status/progress/
# message/workflow_id) plus the report-artifact locations + overall score.
evaluation_runs = Table(
    "evaluation_runs",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("dataset_id", PgUUID(as_uuid=True), nullable=False),
    Column("dataset_version_id", PgUUID(as_uuid=True), nullable=False),
    Column("reference_version_id", PgUUID(as_uuid=True), nullable=True),
    Column("status", String, nullable=False, server_default="pending"),
    Column("progress", Float, nullable=False, server_default="0.0"),
    Column("message", Text, nullable=False, server_default=""),
    Column("workflow_id", String, nullable=True),
    Column("report_uri", String, nullable=True),
    Column("report_html_uri", String, nullable=True),
    Column("overall_score", Float, nullable=True),
    Column("config", JSONB, nullable=False, server_default="{}"),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
)

# Per-expert results for an evaluation run (the mixture-of-experts breakdown).
# Kept in its own table (rather than only inside the JSON artifact) so overall
# scores per dimension are queryable without fetching the object-store report.
evaluation_expert_results = Table(
    "evaluation_expert_results",
    metadata,
    Column("id", PgUUID(as_uuid=True), primary_key=True),
    Column("tenant_id", PgUUID(as_uuid=True), nullable=False),
    Column("run_id", PgUUID(as_uuid=True), nullable=False),
    Column("dimension", String, nullable=False),
    Column("score", Float, nullable=False),
    Column("rationale", Text, nullable=False, server_default=""),
    Column("metrics", JSONB, nullable=False, server_default="{}"),
    Column("recommendations", JSONB, nullable=False, server_default="[]"),
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
    "perturbation_jobs": "tenant_id",
    "image_provider_configs": "tenant_id",
    "video_provider_configs": "tenant_id",
    "audio_provider_configs": "tenant_id",
    "export_artifacts": "tenant_id",
    "evaluation_runs": "tenant_id",
    "evaluation_expert_results": "tenant_id",
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
