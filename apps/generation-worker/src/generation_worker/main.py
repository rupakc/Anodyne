"""Temporal worker process for `GenerationWorkflow`.

Binds the activities in `anodyne_workflows.activities` to real infra
(`SqlDatasetRepository`, `S3ObjectStore`, a Redis progress publisher, and
Ray via `anodyne_compute.ray_init`), then runs a `temporalio` `Worker` on the
"generation" task queue. `build_worker` is the pure wiring step (registers
`GenerationWorkflow` + the five shared activities; testable with fakes, no
live Temporal server needed -- see `tests/test_worker_wiring.py`). `main` is
the process entrypoint: ``python -m generation_worker.main``.

Every modality (tabular/text/image/audio/video) rides the same five activities
via the modality registry in `anodyne_workflows`; the per-modality provider
infra each modality needs is injected here through a single `ActivityContext`.
All provider wiring is gated on a configured `secret_key` -- without one, a
job of that modality fails clearly inside its handler rather than the worker
crashing at startup, and tabular-only deployments need nothing extra.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

import boto3  # type: ignore[import-untyped]
import httpx
import redis.asyncio as redis
from anodyne_audio.registry import SqlAudioProviderRegistry
from anodyne_compute import ray_init
from anodyne_core.ports import SecretStore
from anodyne_dataset.models import DatasetSpec
from anodyne_dataset.ports import (
    AudioProvider,
    DatasetRepository,
    PerturbationRepository,
    Perturbator,
    ProfileRepository,
)
from anodyne_image.registry import SqlImageProviderRegistry
from anodyne_llm.registry import SqlModelRegistry
from anodyne_perturbation import RegistryPerturbator
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import make_engine
from anodyne_storage.secrets import FernetSecretStore
from anodyne_video.adapters.external_api import ReplicateVideoProvider
from anodyne_video.ports import VideoProvider, VideoProviderRegistry
from anodyne_video.registry import SqlVideoProviderRegistry
from anodyne_workflows.activities import (
    ActivityContext,
    ImageConfigRegistry,
    ModelRegistryLike,
    ProgressPublisher,
    assemble_and_upload,
    configure_activities,
    generate_shards,
    plan_shards,
    register_version,
    set_status,
)
from anodyne_workflows.perturbation_activities import (
    PerturbationActivityContext,
    apply_perturbation,
    configure_perturbation_activities,
    register_perturbed_version,
    set_perturbation_status,
)
from anodyne_workflows.perturbation_workflow import PerturbationWorkflow
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

    Every modality-specific field defaults to a value that leaves tabular-only
    wiring/tests unaffected; each is consulted only by its modality's handler.
    """

    repo: DatasetRepository
    s3_bucket: str
    s3_client: Any
    publisher: ProgressPublisher | None = None
    # Tabular from-sample synthesis.
    profile_repo: ProfileRepository | None = None
    ctgan_epochs: int = 100
    enable_sdv: bool = False
    # Text generation.
    model_registry: ModelRegistryLike | None = None
    secret_key: str = ""
    # Image generation.
    image_registry: ImageConfigRegistry | None = None
    secret_store: SecretStore | None = None
    # Audio generation.
    audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] | None = None
    # Video generation.
    video_registry: VideoProviderRegistry | None = None
    video_providers: dict[str, VideoProvider] = field(default_factory=dict)
    # Perturbation (sub-system D). Defaults keep generation-only wiring/tests
    # unaffected; when a `perturbation_repo` is provided the perturbation
    # activities are configured so the same worker serves `PerturbationWorkflow`.
    perturbation_repo: PerturbationRepository | None = None
    perturbator: Perturbator | None = None


def registered_workflows() -> list[type]:
    return [GenerationWorkflow]


def registered_activities() -> list[Callable[..., Any]]:
    return [plan_shards, generate_shards, assemble_and_upload, register_version, set_status]


def registered_perturbation_workflows() -> list[type]:
    return [PerturbationWorkflow]


def registered_perturbation_activities() -> list[Callable[..., Any]]:
    return [set_perturbation_status, apply_perturbation, register_perturbed_version]


def build_worker(client: Client, deps: WorkerDeps) -> Worker:
    """Bind activities to `deps` and construct the `Worker` for the "generation"
    task queue. The worker serves both `GenerationWorkflow` and
    `PerturbationWorkflow` (same queue): both are additive and share infra."""
    configure_activities(
        ActivityContext(
            repo=deps.repo,
            s3_bucket=deps.s3_bucket,
            s3_client=deps.s3_client,
            publisher=deps.publisher,
            profile_repo=deps.profile_repo,
            ctgan_epochs=deps.ctgan_epochs,
            enable_sdv=deps.enable_sdv,
            model_registry=deps.model_registry,
            secret_key=deps.secret_key,
            image_registry=deps.image_registry,
            secret_store=deps.secret_store,
            audio_provider_factory=deps.audio_provider_factory,
            video_registry=deps.video_registry,
            video_providers=deps.video_providers,
        )
    )
    if deps.perturbation_repo is not None:
        configure_perturbation_activities(
            PerturbationActivityContext(
                repo=deps.repo,
                perturbation_repo=deps.perturbation_repo,
                perturbator=deps.perturbator or RegistryPerturbator(),
                s3_bucket=deps.s3_bucket,
                s3_client=deps.s3_client,
                publisher=deps.publisher,
            )
        )
    return Worker(
        client,
        task_queue=TASK_QUEUE,
        workflows=[*registered_workflows(), *registered_perturbation_workflows()],
        activities=[*registered_activities(), *registered_perturbation_activities()],
    )


def _video_providers(secret_store: SecretStore) -> dict[str, VideoProvider]:
    """One adapter per `VideoProviderConfig.provider` value this worker can serve.

    Only the external-API adapter (`ReplicateVideoProvider`) is wired here -- a
    self-hosted/GPU adapter needs a Ray actor holding real model weights, which
    this environment doesn't have. Tenants that register a `provider="self-hosted"`
    config get a clear "no adapter registered" error from the video handler.
    """
    return {
        "replicate": ReplicateVideoProvider(secret_store=secret_store, client=httpx.AsyncClient()),
    }


async def main() -> None:
    settings = get_settings()
    ray_init(settings.ray_address)

    engine = make_engine(settings.database_url)
    repo = SqlDatasetRepository(engine)
    s3_client = boto3.client("s3")
    publisher = RedisProgressPublisher(settings.redis_url)

    # Text/image/audio/video providers all decrypt per-tenant secrets via the
    # Fernet store, so they're only wired when a `secret_key` is configured.
    secret_store: SecretStore | None = None
    model_registry: SqlModelRegistry | None = None
    image_registry: SqlImageProviderRegistry | None = None
    audio_provider_factory: Callable[[DatasetSpec], Awaitable[AudioProvider]] | None = None
    video_registry: SqlVideoProviderRegistry | None = None
    video_providers: dict[str, VideoProvider] = {}
    if settings.secret_key:
        secret_store = FernetSecretStore(settings.secret_key.encode())
        model_registry = SqlModelRegistry(engine, secret_store)
        image_registry = SqlImageProviderRegistry(engine, secret_store)
        audio_provider_factory = AudioProviderFactory(
            SqlAudioProviderRegistry(engine, secret_store), secret_store
        ).build
        video_registry = SqlVideoProviderRegistry(engine, secret_store)
        video_providers = _video_providers(secret_store)

    client = await Client.connect(settings.temporal_address)
    worker = build_worker(
        client,
        WorkerDeps(
            repo=repo,
            s3_bucket=settings.s3_bucket,
            s3_client=s3_client,
            publisher=publisher,
            # `SqlDatasetRepository` implements `DatasetRepository`,
            # `ProfileRepository`, and `PerturbationRepository` -- the same
            # instance serves every role.
            profile_repo=repo,
            perturbation_repo=repo,
            perturbator=RegistryPerturbator(),
            ctgan_epochs=settings.tabular_ctgan_epochs,
            enable_sdv=settings.tabular_enable_sdv,
            model_registry=model_registry,
            secret_key=settings.secret_key,
            image_registry=image_registry,
            secret_store=secret_store,
            audio_provider_factory=audio_provider_factory,
            video_registry=video_registry,
            video_providers=video_providers,
        ),
    )
    await worker.run()


if __name__ == "__main__":
    asyncio.run(main())
