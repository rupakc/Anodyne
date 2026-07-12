from uuid import uuid4

import pytest
import pytest_asyncio
from anodyne_llm.registry import SqlModelRegistry
from anodyne_storage.db import apply_rls, make_engine, metadata
from anodyne_storage.secrets import FernetSecretStore
from cryptography.fernet import Fernet
from sqlalchemy import text
from testcontainers.postgres import PostgresContainer  # type: ignore[import-untyped]

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def engine():  # type: ignore[no-untyped-def]
    with PostgresContainer("postgres:16") as pg:
        eng = make_engine(pg.get_connection_url().replace("psycopg2", "asyncpg"))
        async with eng.begin() as conn:
            await conn.run_sync(metadata.create_all)
            await apply_rls(conn)
            await conn.execute(
                text("CREATE ROLE app LOGIN; GRANT ALL ON ALL TABLES IN SCHEMA public TO app;")
            )
        yield eng
        await eng.dispose()


async def test_create_encrypts_key_and_isolates_tenants(engine):  # type: ignore[no-untyped-def]
    reg = SqlModelRegistry(engine, FernetSecretStore(Fernet.generate_key()))
    t1, t2 = uuid4(), uuid4()
    cfg = await reg.create(
        t1,
        name="c",
        provider="openai",
        model="gpt-4o",
        api_key="sk-secret",
        api_base=None,
        params={},
    )
    assert cfg.secret_ref and cfg.secret_ref != "sk-secret"
    assert await reg.get(t1, cfg.id) is not None
    assert await reg.get(t2, cfg.id) is None  # RLS blocks cross-tenant read
    assert [c.id for c in await reg.list(t1)] == [cfg.id]
    await reg.delete(t1, cfg.id)
    assert await reg.get(t1, cfg.id) is None
