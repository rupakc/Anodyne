"""EvaluationRepository integration test: requires Docker (testcontainers Postgres).

Not run in the default suite. Mirrors `test_dataset_repo.py`: schema/RLS/`app`
role bootstrapped via the superuser connection, sessions under test connect as
the non-superuser `app` role (superusers bypass RLS even with FORCE).
"""

from __future__ import annotations

from uuid import UUID, uuid4

import pytest
import pytest_asyncio
from anodyne_evaluation.models import EvalDimension, EvaluationRun, EvaluationStatus, ExpertScore
from anodyne_evaluation.registry import SqlEvaluationRepository
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


def _run(tid: UUID) -> EvaluationRun:
    return EvaluationRun(
        id=uuid4(),
        tenant_id=tid,
        dataset_id=uuid4(),
        dataset_version_id=uuid4(),
        config={"target_field": "label"},
    )


async def test_run_crud_is_tenant_isolated(engine) -> None:  # type: ignore[no-untyped-def]
    repo = SqlEvaluationRepository(engine)
    t1, t2 = uuid4(), uuid4()
    run = _run(t1)
    await repo.create_run(run)

    got = await repo.get_run(t1, run.id)
    assert got is not None
    assert got.status is EvaluationStatus.PENDING
    assert got.config == {"target_field": "label"}
    assert await repo.get_run(t2, run.id) is None  # RLS + explicit filter

    run.status = EvaluationStatus.SUCCEEDED
    run.overall_score = 0.83
    await repo.save_run(run)
    again = await repo.get_run(t1, run.id)
    assert again is not None
    assert again.status is EvaluationStatus.SUCCEEDED
    assert again.overall_score == 0.83
    assert [r.id for r in await repo.list_runs(t1, run.dataset_id)] == [run.id]


async def test_expert_results_round_trip(engine) -> None:  # type: ignore[no-untyped-def]
    repo = SqlEvaluationRepository(engine)
    t = uuid4()
    run = _run(t)
    await repo.create_run(run)
    scores = [
        ExpertScore(
            dimension=EvalDimension.FIDELITY,
            score=0.9,
            rationale="close",
            metrics={"ks_mean": 0.05},
            recommendations=["keep it up"],
        ),
        ExpertScore(dimension=EvalDimension.PRIVACY, score=0.7, rationale="ok"),
    ]
    await repo.add_expert_results(t, run.id, scores)
    got = {s.dimension: s for s in await repo.get_expert_results(t, run.id)}
    assert got[EvalDimension.FIDELITY].metrics["ks_mean"] == 0.05
    assert got[EvalDimension.FIDELITY].recommendations == ["keep it up"]
    assert got[EvalDimension.PRIVACY].score == 0.7

    # Re-adding replaces (idempotent re-run).
    await repo.add_expert_results(t, run.id, scores[:1])
    assert len(await repo.get_expert_results(t, run.id)) == 1
