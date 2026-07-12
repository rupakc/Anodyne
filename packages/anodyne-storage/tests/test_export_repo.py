"""Export repository integration test: requires Docker (testcontainers Postgres).

Not run in the default suite (`uv run pytest -q -m "not integration"`). Mirrors
`test_dataset_repo.py`'s `engine` fixture bootstrap exactly.
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from anodyne_dataset.models import DatasetSpec, ExportArtifact, FieldSpec, Modality, SemanticType
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import apply_rls, make_engine, metadata
from anodyne_storage.export_repo import SqlExportRepository
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


def _artifact(tid: UUID, dataset_id: UUID, *, fmt: str = "csv") -> ExportArtifact:
    return ExportArtifact(
        id=uuid4(),
        tenant_id=tid,
        dataset_id=dataset_id,
        version_id=uuid4(),
        format=fmt,
        row_count=10,
        object_key=f"datasets/{dataset_id}/export.{fmt}",
    )


async def test_add_and_list_exports_is_tenant_isolated(engine) -> None:  # type: ignore[no-untyped-def]
    dataset_repo = SqlDatasetRepository(engine)
    export_repo = SqlExportRepository(engine)
    t1, t2 = uuid4(), uuid4()
    spec = _spec(t1)
    await dataset_repo.create_spec(spec)

    first = _artifact(t1, spec.id, fmt="csv")
    second = _artifact(t1, spec.id, fmt="parquet")
    await export_repo.add_export(first)
    await export_repo.add_export(second)

    exports = await export_repo.list_exports(t1, spec.id)
    assert {e.id for e in exports} == {first.id, second.id}
    assert await export_repo.list_exports(t2, spec.id) == []  # RLS + explicit filter
