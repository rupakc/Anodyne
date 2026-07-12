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
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import boto3  # type: ignore[import-untyped]
import redis.asyncio as redis
from anodyne_compute import ray_init
from anodyne_core.ports import ObjectStore
from anodyne_dataset.ports import DatasetRepository
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import make_engine
from anodyne_storage.objectstore import S3ObjectStore
from anodyne_workflows.activities import (
    ActivityContext,
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

# The worker process is tenant-agnostic: one `S3ObjectStore` instance backs
# every tenant's activities. Every object key the activities build already
# embeds the tenant/dataset/job path (see
# `anodyne_workflows.activities._shard_key` / `_artifact_key`), so this is
# just a stable outer namespace, not a per-tenant isolation boundary.
_WORKER_STORE_NAMESPACE = uuid.UUID(int=0)


class RedisProgressPublisher:
    """`ProgressPublisher` (see `anodyne_workflows.activities`) backed by Redis pub/sub."""

    def __init__(self, url: str) -> None:
        self._client: redis.Redis = redis.from_url(url)

    async def publish(self, channel: str, message: str) -> None:
        await self._client.publish(channel, message)


@dataclass
class WorkerDeps:
    """Infra the activities need, injected so tests can substitute fakes."""

    repo: DatasetRepository
    object_store: ObjectStore
    publisher: ProgressPublisher | None = None


def registered_workflows() -> list[type]:
    return [GenerationWorkflow]


def registered_activities() -> list[Callable[..., Any]]:
    return [plan_shards, generate_shards, assemble_and_upload, register_version, set_status]


def build_worker(client: Client, deps: WorkerDeps) -> Worker:
    """Bind activities to `deps` and construct the `Worker` for the "generation" task queue."""
    configure_activities(
        ActivityContext(repo=deps.repo, object_store=deps.object_store, publisher=deps.publisher)
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
    object_store = S3ObjectStore(settings.s3_bucket, _WORKER_STORE_NAMESPACE, client=s3_client)
    publisher = RedisProgressPublisher(settings.redis_url)

    client = await Client.connect(settings.temporal_address)
    worker = build_worker(
        client, WorkerDeps(repo=repo, object_store=object_store, publisher=publisher)
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
