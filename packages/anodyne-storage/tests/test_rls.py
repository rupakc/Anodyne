"""RLS tenant-isolation test: requires Docker (testcontainers Postgres).

Not run in the default suite (`uv run pytest -q -m "not integration"`); it
needs a real Postgres to enforce row-level security, which SQLite/mocks
cannot emulate. See CI notes in task-7-report.md.

Note: the schema, policies, and `app` role are created via the bootstrap
(superuser) connection, but the sessions under test connect as the
non-superuser `app` role. This is deliberate — superusers BYPASS row-level
security even when the table has ``FORCE ROW LEVEL SECURITY``, so testing
isolation through a superuser connection would not exercise RLS at all.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import pytest_asyncio
from anodyne_storage.db import apply_rls, make_engine, metadata, tenant_session
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def engine():  # type: ignore[no-untyped-def]
    with PostgresContainer("postgres:16") as pg:
        admin_dsn = pg.get_connection_url().replace("psycopg2", "asyncpg")
        # Bootstrap as the superuser: create schema, RLS policies, and a
        # non-superuser login role that IS subject to RLS.
        admin_eng = make_engine(admin_dsn)
        async with admin_eng.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await apply_rls(conn)
            await conn.execute(text("CREATE ROLE app LOGIN PASSWORD 'app'"))
            await conn.execute(text("GRANT USAGE ON SCHEMA public TO app"))
            await conn.execute(text("GRANT ALL ON ALL TABLES IN SCHEMA public TO app"))
        await admin_eng.dispose()

        # Sessions under test connect as `app` so row-level security applies.
        app_dsn = admin_dsn.replace(f"//{pg.username}:{pg.password}@", "//app:app@")
        app_eng = make_engine(app_dsn)
        yield app_eng
        await app_eng.dispose()


async def test_tenant_isolation(engine) -> None:  # type: ignore[no-untyped-def]
    t1, t2 = uuid4(), uuid4()
    async with tenant_session(engine, t1) as s:
        await s.execute(
            text(
                "INSERT INTO tenants (id, name, org_ref, status) VALUES (:id,'A','orgA','active')"
            ),
            {"id": t1},
        )
        await s.commit()
    # tenant 2's session must NOT see tenant 1's row
    async with tenant_session(engine, t2) as s:
        rows = (await s.execute(text("SELECT id FROM tenants"))).all()
        assert rows == []
    async with tenant_session(engine, t1) as s:
        rows = (await s.execute(text("SELECT id FROM tenants"))).all()
        assert len(rows) == 1


async def test_tenant_isolation_hitl_tables(engine) -> None:  # type: ignore[no-untyped-def]
    """Sub-system G's three new tables (`annotations`, `feedback`,
    `review_tasks`) get the same per-tenant RLS policy as every other tenant
    table -- verified the same way as `test_tenant_isolation` above."""
    t1, t2 = uuid4(), uuid4()
    dataset_id, version_id, review_id = uuid4(), uuid4(), uuid4()
    async with tenant_session(engine, t1) as s:
        await s.execute(
            text(
                "INSERT INTO annotations (id, tenant_id, dataset_id, version_id, author) "
                "VALUES (:id, :tid, :did, :vid, 'u@x.io')"
            ),
            {"id": uuid4(), "tid": t1, "did": dataset_id, "vid": version_id},
        )
        await s.execute(
            text(
                "INSERT INTO feedback (id, tenant_id, target_type, target_id, author) "
                "VALUES (:id, :tid, 'dataset_version', :target, 'u@x.io')"
            ),
            {"id": uuid4(), "tid": t1, "target": version_id},
        )
        await s.execute(
            text(
                "INSERT INTO review_tasks (id, tenant_id, kind, target_type, target_id) "
                "VALUES (:id, :tid, 'schema_approval', 'dataset', :target)"
            ),
            {"id": review_id, "tid": t1, "target": dataset_id},
        )
        await s.commit()

    for table in ("annotations", "feedback", "review_tasks"):
        async with tenant_session(engine, t2) as s:
            rows = (await s.execute(text(f"SELECT id FROM {table}"))).all()
            assert rows == [], f"tenant 2 should not see tenant 1's {table} rows"
        async with tenant_session(engine, t1) as s:
            rows = (await s.execute(text(f"SELECT id FROM {table}"))).all()
            assert len(rows) == 1
