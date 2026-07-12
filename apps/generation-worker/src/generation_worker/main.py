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
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

import boto3  # type: ignore[import-untyped]
import redis.asyncio as redis
from anodyne_compute import ray_init
from anodyne_dataset.models import DatasetSpec
from anodyne_dataset.ports import AudioProvider, DatasetRepository
from anodyne_llm.registry import SqlModelRegistry
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import make_engine
from anodyne_storage.secrets import FernetSecretStore
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

from generation_worker.audio import AudioProviderFactory
from generation_worker.config import get_settings

TASK_QUEUE = "generation"


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
    # Resolves a tenant's registered ModelConfig into a concrete AudioProvider
    # (see `generation_worker.audio.AudioProviderFactory`). `None` disables
    # the audio path (any `Modality.AUDIO` job then fails clearly in
    # `_generate_audio_shards` rather than silently no-oping).
    audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] | None = None


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
            audio_provider_factory=deps.audio_provider_factory,
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

    # The audio path needs a Fernet secret key to decrypt provider API keys
    # (exactly like the gateway's `get_model_registry`); without one, audio
    # generation jobs fail clearly at `_generate_audio_shards` rather than
    # silently no-oping. Tabular generation is unaffected either way.
    audio_provider_factory = None
    if settings.secret_key:
        secret_store = FernetSecretStore(settings.secret_key.encode())
        model_registry = SqlModelRegistry(engine, secret_store)
        audio_provider_factory = AudioProviderFactory(model_registry, secret_store).build

    client = await Client.connect(settings.temporal_address)
    worker = build_worker(
        client,
        WorkerDeps(
            repo=repo,
            s3_bucket=settings.s3_bucket,
            s3_client=s3_client,
            publisher=publisher,
            audio_provider_factory=audio_provider_factory,
        ),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
