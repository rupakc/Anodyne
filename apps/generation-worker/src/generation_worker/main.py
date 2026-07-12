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
from dataclasses import dataclass, field
from typing import Any

import boto3  # type: ignore[import-untyped]
import httpx
import redis.asyncio as redis
from anodyne_compute import ray_init
from anodyne_core.ports import SecretStore
from anodyne_dataset.ports import DatasetRepository
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import make_engine
from anodyne_storage.secrets import FernetSecretStore
from anodyne_video.adapters.external_api import ReplicateVideoProvider
from anodyne_video.ports import VideoProvider, VideoProviderRegistry
from anodyne_video.registry import SqlVideoProviderRegistry
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
from anodyne_workflows.video_activities import (
    VideoActivityContext,
    assemble_video_manifest,
    configure_video_activities,
    generate_video_items,
    plan_video_items,
    register_video_version,
)
from anodyne_workflows.workflow import GenerationWorkflow
from temporalio.client import Client
from temporalio.worker import Worker

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
    # Video-modality deps: optional so existing tabular-only construction
    # keeps working unchanged. `video_registry=None` configures the video
    # activities with an empty provider registry/adapter set -- fine, since a
    # deployment that never runs video jobs never invokes them; one that does
    # gets a clear runtime error from `generate_video_items` rather than a
    # crash at worker startup.
    video_registry: VideoProviderRegistry | None = None
    video_providers: dict[str, VideoProvider] = field(default_factory=dict)


def registered_workflows() -> list[type]:
    return [GenerationWorkflow]


def registered_activities() -> list[Callable[..., Any]]:
    return [
        plan_shards,
        generate_shards,
        assemble_and_upload,
        register_version,
        set_status,
        plan_video_items,
        generate_video_items,
        assemble_video_manifest,
        register_video_version,
    ]


class _EmptyVideoProviderRegistry(VideoProviderRegistry):
    """Placeholder used when no video registry is configured (tabular-only deployments)."""

    async def create(
        self,
        tenant_id: Any,
        *,
        name: str,
        provider: str,
        model: str,
        api_key: str | None,
        api_base: str | None,
        params: dict[str, object],
    ) -> Any:
        raise NotImplementedError("video provider registry is not configured on this worker")

    async def get(self, tenant_id: Any, config_id: Any) -> Any:
        return None

    async def list(self, tenant_id: Any) -> list[Any]:
        return []

    async def delete(self, tenant_id: Any, config_id: Any) -> None:
        return None


def build_worker(client: Client, deps: WorkerDeps) -> Worker:
    """Bind activities to `deps` and construct the `Worker` for the "generation" task queue."""
    configure_activities(
        ActivityContext(
            repo=deps.repo,
            s3_bucket=deps.s3_bucket,
            s3_client=deps.s3_client,
            publisher=deps.publisher,
        )
    )
    configure_video_activities(
        VideoActivityContext(
            repo=deps.repo,
            s3_bucket=deps.s3_bucket,
            s3_client=deps.s3_client,
            video_registry=deps.video_registry or _EmptyVideoProviderRegistry(),
            providers=deps.video_providers,
        )
    )
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=registered_workflows(),
        activities=registered_activities(),
    )


def _video_deps(
    settings: Any, secret_store: SecretStore, engine: Any
) -> tuple[VideoProviderRegistry, dict[str, VideoProvider]]:
    """Real video infra: a SQL-backed provider registry + one adapter per
    `VideoProviderConfig.provider` value this worker knows how to serve.

    Only the external-API adapter (`ReplicateVideoProvider`) is wired here --
    a self-hosted/GPU adapter needs a Ray actor holding real model weights
    (see `anodyne_video.adapters.self_hosted`'s docstring), which this
    environment doesn't have. Tenants that register a `provider="self-hosted"`
    config will get a clear "no adapter registered" error from
    `generate_video_items` until that's wired up.
    """
    registry = SqlVideoProviderRegistry(engine, secret_store)
    providers: dict[str, VideoProvider] = {
        "replicate": ReplicateVideoProvider(secret_store=secret_store, client=httpx.AsyncClient()),
    }
    return registry, providers


async def main() -> None:
    settings = get_settings()
    ray_init(settings.ray_address)

    engine = make_engine(settings.database_url)
    repo = SqlDatasetRepository(engine)
    s3_client = boto3.client("s3")
    publisher = RedisProgressPublisher(settings.redis_url)
    secret_store = FernetSecretStore(settings.secret_key.encode())
    video_registry, video_providers = _video_deps(settings, secret_store, engine)

    client = await Client.connect(settings.temporal_address)
    worker = build_worker(
        client,
        WorkerDeps(
            repo=repo,
            s3_bucket=settings.s3_bucket,
            s3_client=s3_client,
            publisher=publisher,
            video_registry=video_registry,
            video_providers=video_providers,
        ),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
