"""Unit tests for the image-modality branches added to
`anodyne_workflows.activities` (`generate_shards`/`assemble_and_upload`/
`register_version`) and `anodyne_workflows.image_activities`.

No Temporal, no live Ray, no network/GPU: the Ray dispatch itself
(`remote_generate_image_shard.remote` + `ray.get`) is monkeypatched to run
the same shard-generation code inline (a fake `ImageProvider` registered
into `anodyne_image.factory` for the duration of the test), mirroring how
`test_activities.py` exercises `assemble_and_upload`/`register_version`
against a real (moto-mocked) S3 bucket without touching `generate_shards`'s
real Ray path.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from typing import Any
from uuid import UUID

import boto3  # type: ignore[import-untyped]
import pytest
from anodyne_compute.image_tasks import generate_image_shard_bytes
from anodyne_core.models import ModelConfig
from anodyne_dataset.models import (
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    GenerationJob,
    Modality,
    SemanticType,
)
from anodyne_dataset.ports import DatasetRepository
from anodyne_image.factory import _REGISTRY, register_provider
from anodyne_image.models import GeneratedImage
from anodyne_image.ports import ImageProvider
from anodyne_workflows import image_activities
from anodyne_workflows.activities import (
    ActivityContext,
    assemble_and_upload,
    configure_activities,
    generate_shards,
    register_version,
)
from anodyne_workflows.workflow import GenerationInput
from moto import mock_aws

_BUCKET = "test-bucket"
_FAKE_PROVIDER = "fake-test-image-provider"


@pytest.fixture
def s3_client() -> Generator[Any, None, None]:
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket=_BUCKET)
        yield c


class _FakeImageProvider(ImageProvider):
    async def generate(self, prompt: str, *, seed: int, size: str = "1024x1024") -> GeneratedImage:
        return GeneratedImage(data=f"{prompt}|{seed}".encode())


@pytest.fixture(autouse=True)
def _register_fake_image_provider() -> Generator[None, None, None]:
    register_provider(_FAKE_PROVIDER, lambda cfg, key: _FakeImageProvider())
    yield
    del _REGISTRY[_FAKE_PROVIDER]


@pytest.fixture(autouse=True)
def _bypass_real_ray(monkeypatch: pytest.MonkeyPatch) -> None:
    """`generate_image_shards` dispatches via `remote_generate_image_shard.remote`
    (real Ray: instant task submission) + `ray.get` (blocks, off the event
    loop thread via `asyncio.to_thread`). The fakes below preserve that shape
    -- `.remote()` just packages the args, and the actual (synchronous, uses
    its own `asyncio.run`) generation happens inside `ray.get`, which the
    real code already runs via `asyncio.to_thread` -- so this never collides
    with pytest-asyncio's running event loop, exactly like a real Ray worker
    process (which has no event loop of its own) wouldn't.
    """

    class _FakeRemote:
        def remote(
            self,
            spec: DatasetSpec,
            start: int,
            count: int,
            seed: int,
            provider_config: ModelConfig,
            api_key: str | None,
        ) -> tuple[DatasetSpec, int, int, int, ModelConfig, str | None]:
            return (spec, start, count, seed, provider_config, api_key)

    class _FakeRay:
        @staticmethod
        def get(
            ref: tuple[DatasetSpec, int, int, int, ModelConfig, str | None],
        ) -> bytes:
            return generate_image_shard_bytes(*ref)

    monkeypatch.setattr(image_activities, "remote_generate_image_shard", _FakeRemote())
    monkeypatch.setattr(image_activities, "ray", _FakeRay())


class _FakeImageRegistry:
    def __init__(self, configs: list[ModelConfig]) -> None:
        self._configs = configs

    async def list(self, tenant_id: UUID) -> list[ModelConfig]:
        return self._configs


class _FakeDatasetRepository(DatasetRepository):
    def __init__(self, spec: DatasetSpec | None = None) -> None:
        self.spec = spec
        self.versions: list[DatasetVersion] = []
        self.jobs: dict[UUID, GenerationJob] = {}

    async def create_spec(self, spec: DatasetSpec) -> None: ...

    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        return self.spec

    async def list_specs(self, tenant_id: UUID) -> list[DatasetSpec]:
        return []

    async def update_spec(self, spec: DatasetSpec) -> None: ...

    async def save_job(self, job: GenerationJob) -> None:
        self.jobs[job.id] = job

    async def get_job(self, tenant_id: UUID, job_id: UUID) -> GenerationJob | None:
        return self.jobs.get(job_id)

    async def add_version(self, version: DatasetVersion) -> None:
        self.versions.append(version)

    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]:
        return []


def _image_spec(tenant_id: UUID, dataset_id: UUID) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,
        tenant_id=tenant_id,
        name="d",
        description="a widget",
        modality=Modality.IMAGE,
        source="description",
        fields=[
            FieldSpec(
                name="label",
                semantic_type=SemanticType.CATEGORICAL,
                constraints={"choices": ["a", "b"]},
            )
        ],
        target_rows=6,
    )


def _fake_provider_config(tenant_id: UUID, secret_ref: str | None = None) -> ModelConfig:
    return ModelConfig(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        name="p",
        provider=_FAKE_PROVIDER,
        model="m",
        secret_ref=secret_ref,
    )


def _input(job_id: UUID, tenant_id: UUID, dataset_id: UUID) -> GenerationInput:
    return GenerationInput(
        job_id=str(job_id),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=6,
        seed=1,
        modality="image",
    )


async def test_generate_shards_dispatches_image_path_and_uploads_shards(s3_client: Any) -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _image_spec(tenant_id, dataset_id)
    repo = _FakeDatasetRepository(spec=spec)
    registry = _FakeImageRegistry([_fake_provider_config(tenant_id)])
    configure_activities(
        ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=s3_client, image_registry=registry)
    )
    inp = _input(job_id, tenant_id, dataset_id)

    keys = await generate_shards(inp, [[0, 3], [3, 3]])

    assert len(keys) == 2
    for key in keys:
        obj = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{key}")
        assert obj["Body"].read()  # non-empty parquet bytes


async def test_generate_shards_image_path_requires_a_registered_provider(s3_client: Any) -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _image_spec(tenant_id, dataset_id)
    repo = _FakeDatasetRepository(spec=spec)
    configure_activities(
        ActivityContext(
            repo=repo, s3_bucket=_BUCKET, s3_client=s3_client, image_registry=_FakeImageRegistry([])
        )
    )
    inp = _input(job_id, tenant_id, dataset_id)

    with pytest.raises(ValueError, match="no image provider configured"):
        await generate_shards(inp, [[0, 6]])


async def test_full_image_pipeline_produces_manifest_and_image_files(s3_client: Any) -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _image_spec(tenant_id, dataset_id)
    repo = _FakeDatasetRepository(spec=spec)
    registry = _FakeImageRegistry([_fake_provider_config(tenant_id)])
    configure_activities(
        ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=s3_client, image_registry=registry)
    )
    inp = _input(job_id, tenant_id, dataset_id)

    shard_keys = await generate_shards(inp, [[0, 3], [3, 3]])
    manifest_key = await assemble_and_upload(inp, shard_keys)
    await register_version(inp, manifest_key, rows=6)

    assert manifest_key == f"datasets/{dataset_id}/{job_id}/manifest.json"
    manifest_obj = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{manifest_key}")
    manifest = json.loads(manifest_obj["Body"].read())
    items = manifest["items"]
    assert [item["item_index"] for item in items] == [0, 1, 2, 3, 4, 5]
    for item in items:
        assert item["prompt"]
        assert item["label"] in ("a", "b")
        image_obj = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{item['object_key']}")
        assert image_obj["Body"].read()  # the individual image file exists

    assert len(repo.versions) == 1
    version = repo.versions[0]
    assert version.artifact_uri == manifest_key
    assert version.format == "image_manifest"
    assert version.row_count == 6


async def test_generate_shards_image_path_decrypts_secret(s3_client: Any) -> None:
    """The provider's `api_key` argument must be the *decrypted* secret, not
    the encrypted `secret_ref` -- proven via a fake SecretStore that records
    what it was asked to decrypt.
    """

    class _RecordingSecretStore:
        def __init__(self) -> None:
            self.decrypted: list[str] = []

        def encrypt(self, plaintext: str) -> str:
            return f"enc:{plaintext}"

        def decrypt(self, ref: str) -> str:
            self.decrypted.append(ref)
            return ref.removeprefix("enc:")

    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _image_spec(tenant_id, dataset_id)
    repo = _FakeDatasetRepository(spec=spec)
    secret_store = _RecordingSecretStore()
    registry = _FakeImageRegistry([_fake_provider_config(tenant_id, secret_ref="enc:sk-real-key")])
    configure_activities(
        ActivityContext(
            repo=repo,
            s3_bucket=_BUCKET,
            s3_client=s3_client,
            image_registry=registry,
            secret_store=secret_store,  # type: ignore[arg-type]
        )
    )
    inp = _input(job_id, tenant_id, dataset_id)

    await generate_shards(inp, [[0, 2]])

    assert secret_store.decrypted == ["enc:sk-real-key"]
