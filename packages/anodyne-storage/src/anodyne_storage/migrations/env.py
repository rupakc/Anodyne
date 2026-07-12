from __future__ import annotations

import os
from logging.config import fileConfig

from alembic import context
from anodyne_storage.db import metadata
from sqlalchemy import Connection, pool
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

# Alembic Config object, giving access to values in alembic.ini.
config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = metadata

# Prefer an explicit DSN from the environment over the ini fallback so the
# same migration can run against dev/CI/prod without editing alembic.ini.
DSN = os.environ.get("ANODYNE_DB_DSN") or config.get_main_option("sqlalchemy.url")


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection."""
    context.configure(
        url=DSN,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_migrations_online() -> None:
    """Run migrations against a live DB using an async engine.

    Alembic's migration runner is sync-only, so the async connection is
    bridged with `run_sync`, which executes `do_run_migrations` on a worker
    thread bound to this connection.
    """
    assert DSN is not None, "ANODYNE_DB_DSN must be set (or alembic.ini sqlalchemy.url configured)"
    connectable: AsyncEngine = create_async_engine(DSN, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


if context.is_offline_mode():
    run_migrations_offline()
else:
    import asyncio

    asyncio.run(run_migrations_online())
