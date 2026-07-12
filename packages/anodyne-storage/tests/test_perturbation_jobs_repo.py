"""Perturbation-job repository integration test: requires Docker (Postgres).

Mirrors `test_dataset_repo.py`: schema/RLS/`app` role bootstrapped as superuser,
sessions under test connect as the non-superuser `app` role so RLS applies.
Not run in the default suite (`-m "not integration"`).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    JobStatus,
    Modality,
    PerturbationFamily,
    PerturbationJob,
    PerturbationSpec,
    SemanticType,
)
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


async def test_version_lineage_roundtrip(engine) -> None:  # type: ignore[no-untyped-def]
    repo = SqlDatasetRepository(engine)
    t = uuid4()
    s = _spec(t)
    await repo.create_spec(s)
    parent = DatasetVersion(id=uuid4(), tenant_id=t, dataset_id=s.id, artifact_uri="a")
    child = DatasetVersion(
        id=uuid4(),
        tenant_id=t,
        dataset_id=s.id,
        artifact_uri="b",
        parent_version_id=parent.id,
    )
    await repo.add_version(parent)
    await repo.add_version(child)
    versions = {v.id: v for v in await repo.list_versions(t, s.id)}
    assert versions[parent.id].parent_version_id is None
    assert versions[child.id].parent_version_id == parent.id


async def test_perturbation_job_roundtrip_is_tenant_isolated(engine) -> None:  # type: ignore[no-untyped-def]
    repo = SqlDatasetRepository(engine)
    t1, t2 = uuid4(), uuid4()
    s = _spec(t1)
    await repo.create_spec(s)
    job = PerturbationJob(
        id=uuid4(),
        tenant_id=t1,
        dataset_id=s.id,
        parent_version_id=uuid4(),
        spec=PerturbationSpec(
            family=PerturbationFamily.BIAS,
            intensity=0.5,
            target_fields=["age"],
            params={"target_ratio": 0.9},
            seed=42,
        ),
    )
    await repo.save_perturbation_job(job)

    got = await repo.get_perturbation_job(t1, job.id)
    assert got is not None
    assert got.spec.family is PerturbationFamily.BIAS
    assert got.spec.target_fields == ["age"]
    assert got.spec.params == {"target_ratio": 0.9}
    assert got.spec.seed == 42  # seed round-trips so the stored job is replayable
    assert await repo.get_perturbation_job(t2, job.id) is None  # RLS + filter
    assert [j.id for j in await repo.list_perturbation_jobs(t1, s.id)] == [job.id]


async def test_perturbation_job_result_version_upsert(engine) -> None:  # type: ignore[no-untyped-def]
    repo = SqlDatasetRepository(engine)
    t = uuid4()
    s = _spec(t)
    await repo.create_spec(s)
    job = PerturbationJob(
        id=uuid4(),
        tenant_id=t,
        dataset_id=s.id,
        parent_version_id=uuid4(),
        spec=PerturbationSpec(family=PerturbationFamily.NOISE),
    )
    await repo.save_perturbation_job(job)
    result_id = uuid4()
    job.result_version_id = result_id
    job.status = JobStatus.SUCCEEDED
    await repo.save_perturbation_job(job)
    got = await repo.get_perturbation_job(t, job.id)
    assert got is not None
    assert got.result_version_id == result_id
