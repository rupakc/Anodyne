"""Registry integration test: requires Docker (testcontainers Postgres).

Mirrors `packages/anodyne-llm/tests/test_registry.py` exactly, over the
separate `image_provider_configs` table.
"""

from __future__ import annotations

from uuid import uuid4

import pytest
import pytest_asyncio
from anodyne_image.registry import SqlImageProviderRegistry
from anodyne_storage.db import apply_rls, make_engine, metadata
from anodyne_storage.secrets import FernetSecretStore
from cryptography.fernet import Fernet
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


async def test_create_encrypts_key_and_isolates_tenants(engine):  # type: ignore[no-untyped-def]
    reg = SqlImageProviderRegistry(engine, FernetSecretStore(Fernet.generate_key()))
    t1, t2 = uuid4(), uuid4()
    cfg = await reg.create(
        t1,
        name="my-openai",
        provider="openai-images",
        model="dall-e-3",
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


async def test_self_hosted_provider_needs_no_key(engine):  # type: ignore[no-untyped-def]
    reg = SqlImageProviderRegistry(engine, FernetSecretStore(Fernet.generate_key()))
    tenant_id = uuid4()
    cfg = await reg.create(
        tenant_id,
        name="gpu-node",
        provider="sdxl-self-hosted",
        model="stabilityai/stable-diffusion-xl-base-1.0",
        api_key=None,
        api_base=None,
        params={},
    )
    assert cfg.secret_ref is None
