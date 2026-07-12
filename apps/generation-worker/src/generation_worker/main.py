"""Temporal worker process for `GenerationWorkflow`.

Binds the activities in `anodyne_workflows.activities` to real infra
(`SqlDatasetRepository`, `S3ObjectStore`, a Redis progress publisher, and
Ray via `anodyne_compute.ray_init`), then runs a `temporalio` `Worker` on the
"generation" task queue. `build_worker` is the pure wiring step (registers
`GenerationWorkflow` + the five activities; testable with fakes, no live
Temporal server needed — see `tests/test_worker_wiring.py`). `main` is the
process entrypoint: ``python -m generation_worker.main``.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import boto3  # type: ignore[import-untyped]
import redis.asyncio as redis
from anodyne_compute import ray_init
from anodyne_dataset.ports import DatasetRepository
from anodyne_llm.registry import SqlModelRegistry
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import make_engine
from anodyne_storage.secrets import FernetSecretStore
from anodyne_workflows.activities import (
    ActivityContext,
    ModelRegistryLike,
    ProgressPublisher,
    assemble_and_upload,
    configure_activities,
    generate_shards,
    plan_shards,
    register_version,
    set_status,
)
from anodyne_workflows.workflow import GenerationWorkflow
from temporalio.client import Client
from temporalio.worker import Worker

from generation_worker.config import get_settings

TASK_QUEUE = "generation"


class SecretStoreConfigError(RuntimeError):
    """Raised when `ANODYNE_SECRET_KEY` is missing or not a valid Fernet key."""


def _secret_store(secret_key: str) -> FernetSecretStore:
    # Mirrors `api_gateway.deps._secret_store` (not imported -- an app should
    # not depend on a sibling app). Text generation needs the tenant's model
    # registry, which needs a secret store even though `register_version`'s
    # `.get()` path never itself decrypts (see `ActivityContext` docstring).
    try:
        return FernetSecretStore(secret_key.encode())
    except ValueError as exc:
        raise SecretStoreConfigError(
            "ANODYNE_SECRET_KEY is missing or not a valid Fernet key. Generate one with: "
            'python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())" and set it in your .env.'
        ) from exc


class RedisProgressPublisher:
    """`ProgressPublisher` (see `anodyne_workflows.activities`) backed by Redis pub/sub."""

    def __init__(self, url: str) -> None:
        self._client: redis.Redis = redis.from_url(url)

    async def publish(self, channel: str, message: str) -> None:
        await self._client.publish(channel, message)


@dataclass
class WorkerDeps:
    """Infra the activities need, injected so tests can substitute fakes.

    `s3_bucket` + `s3_client` (not a pre-built `ObjectStore`): each activity
    builds its own tenant-scoped `S3ObjectStore` from `GenerationInput.tenant_id`
    (see `anodyne_workflows.activities._object_store`). A single pre-built,
    tenant-agnostic store here was the root cause of the write/read prefix
    mismatch that made every download 404.
    """

    repo: DatasetRepository
    s3_bucket: str
    s3_client: Any
    publisher: ProgressPublisher | None = None
    # Only consulted for modality="text" datasets -- see
    # `anodyne_workflows.activities.ActivityContext`.
    model_registry: ModelRegistryLike | None = None
    secret_key: str = ""


def registered_workflows() -> list[type]:
    return [GenerationWorkflow]


def registered_activities() -> list[Callable[..., Any]]:
    return [plan_shards, generate_shards, assemble_and_upload, register_version, set_status]


def build_worker(client: Client, deps: WorkerDeps) -> Worker:
    """Bind activities to `deps` and construct the `Worker` for the "generation" task queue."""
    configure_activities(
        ActivityContext(
            repo=deps.repo,
            s3_bucket=deps.s3_bucket,
            s3_client=deps.s3_client,
            publisher=deps.publisher,
            model_registry=deps.model_registry,
            secret_key=deps.secret_key,
        )
    )
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=registered_workflows(),
        activities=registered_activities(),
    )


async def main() -> None:
    settings = get_settings()
    ray_init(settings.ray_address)

    engine = make_engine(settings.database_url)
    repo = SqlDatasetRepository(engine)
    s3_client = boto3.client("s3")
    publisher = RedisProgressPublisher(settings.redis_url)
    model_registry = SqlModelRegistry(engine, _secret_store(settings.secret_key))

    client = await Client.connect(settings.temporal_address)
    worker = build_worker(
        client,
        WorkerDeps(
            repo=repo,
            s3_bucket=settings.s3_bucket,
            s3_client=s3_client,
            publisher=publisher,
            model_registry=model_registry,
            secret_key=settings.secret_key,
        ),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
