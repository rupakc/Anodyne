"""Dataset repository integration test: requires Docker (testcontainers Postgres).

Not run in the default suite (`uv run pytest -q -m "not integration"`).

Mirrors `test_rls.py`: schema/RLS/`app` role are bootstrapped via the
superuser connection, but the sessions under test connect as the
non-superuser `app` role, since superusers bypass RLS even with
``FORCE ROW LEVEL SECURITY``.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from anodyne_dataset.models import DatasetSpec, FieldSpec, GenerationJob, Modality, SemanticType
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import apply_rls, make_engine, metadata
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def engine():  # type: ignore[no-untyped-def]
    with PostgresContainer("postgres:16") as pg:
        admin_dsn = pg.get_connection_url().replace("psycopg2", "asyncpg")
        admin_eng = make_engine(admin_dsn)
        async with admin_eng.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await apply_rls(conn)
            await conn.execute(text("CREATE ROLE app LOGIN PASSWORD 'app'"))
            await conn.execute(text("GRANT USAGE ON SCHEMA public TO app"))
            await conn.execute(text("GRANT ALL ON ALL TABLES IN SCHEMA public TO app"))
        await admin_eng.dispose()

        app_dsn = admin_dsn.replace(f"//{pg.username}:{pg.password}@", "//app:app@")
        app_eng = make_engine(app_dsn)
        yield app_eng
        await app_eng.dispose()


def _spec(tid: UUID) -> DatasetSpec:
    return DatasetSpec(
        id=uuid4(),
        tenant_id=tid,
        name="d",
        description="x",
        modality=Modality.TABULAR,
        source="description",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=10,
    )


async def test_spec_crud_is_tenant_isolated(engine) -> None:  # type: ignore[no-untyped-def]
    repo = SqlDatasetRepository(engine)
    t1, t2 = uuid4(), uuid4()
    s = _spec(t1)
    await repo.create_spec(s)
    got = await repo.get_spec(t1, s.id)
    assert got is not None
    assert got.name == "d"
    assert got.created_at == s.created_at
    assert await repo.get_spec(t2, s.id) is None  # RLS + explicit filter
    assert [x.id for x in await repo.list_specs(t1)] == [s.id]


async def test_job_roundtrip(engine) -> None:  # type: ignore[no-untyped-def]
    repo = SqlDatasetRepository(engine)
    t = uuid4()
    s = _spec(t)
    await repo.create_spec(s)
    j = GenerationJob(id=uuid4(), tenant_id=t, dataset_id=s.id)
    await repo.save_job(j)
    got = await repo.get_job(t, j.id)
    assert got is not None
    assert got.dataset_id == s.id
