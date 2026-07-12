"""Tests for the `spec.modality`-keyed video dispatch in `anodyne_workflows.activities`.

Video now rides the shared `plan_shards`/`generate_shards`/`assemble_and_upload`/
`register_version` activities via the modality registry (its `VideoHandler`),
rather than a separate `video_activities` module -- so these tests exercise the
same public entry points as tabular/audio, with a video-configured
`ActivityContext`. A mocked `VideoProvider` throughout (no GPU, no network).
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from typing import Any

import boto3  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import DatasetSpec, DatasetVersion, GenerationJob, Modality
from anodyne_dataset.ports import DatasetRepository
from anodyne_video.models import VideoAsset, VideoGenerationRequest, VideoProviderConfig
from anodyne_video.ports import VideoProvider, VideoProviderRegistry
from anodyne_workflows.activities import (
    ActivityContext,
    assemble_and_upload,
    configure_activities,
    generate_shards,
    plan_shards,
    register_version,
)
from anodyne_workflows.workflow import GenerationInput
from moto import mock_aws

_BUCKET = "test-video-bucket"


@pytest.fixture
def s3_client() -> Generator[Any, None, None]:
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket=_BUCKET)
        yield c


class _FakeDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self.specs: dict[uuid.UUID, DatasetSpec] = {}
        self.versions: list[DatasetVersion] = []

    async def create_spec(self, spec: DatasetSpec) -> None:
        self.specs[spec.id] = spec

    async def get_spec(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> DatasetSpec | None:
        return self.specs.get(dataset_id)

    async def list_specs(self, tenant_id: uuid.UUID) -> list[DatasetSpec]:
        return list(self.specs.values())

    async def update_spec(self, spec: DatasetSpec) -> None: ...

    async def save_job(self, job: GenerationJob) -> None: ...

    async def get_job(self, tenant_id: uuid.UUID, job_id: uuid.UUID) -> GenerationJob | None:
        return None

    async def add_version(self, version: DatasetVersion) -> None:
        self.versions.append(version)

    async def list_versions(
        self, tenant_id: uuid.UUID, dataset_id: uuid.UUID
    ) -> list[DatasetVersion]:
        return []


class _FakeVideoProvider(VideoProvider):
    async def generate(
        self, config: VideoProviderConfig, request: VideoGenerationRequest
    ) -> VideoAsset:
        return VideoAsset(
            content=f"clip-{request.seed}".encode(),
            duration_seconds=request.duration_seconds,
            width=request.width,
            height=request.height,
            fps=request.fps,
            seed=request.seed,
            provider=config.provider,
            model=config.model,
        )


class _FakeVideoProviderRegistry(VideoProviderRegistry):
    def __init__(self, configs: list[VideoProviderConfig]) -> None:
        self._configs = configs

    async def create(
        self,
        tenant_id: uuid.UUID,
        *,
        name: str,
        provider: str,
        model: str,
        api_key: str | None,
        api_base: str | None,
        params: dict[str, object],
    ) -> VideoProviderConfig:
        raise NotImplementedError

    async def get(self, tenant_id: uuid.UUID, config_id: uuid.UUID) -> VideoProviderConfig | None:
        return next((c for c in self._configs if c.id == config_id), None)

    async def list(self, tenant_id: uuid.UUID) -> list[VideoProviderConfig]:
        return self._configs

    async def delete(self, tenant_id: uuid.UUID, config_id: uuid.UUID) -> None: ...


def _config(*, enabled: bool = True, tenant_id: uuid.UUID | None = None) -> VideoProviderConfig:
    return VideoProviderConfig(
        id=uuid.uuid4(),
        tenant_id=tenant_id or uuid.uuid4(),
        name="c",
        provider="fake",
        model="fake-model",
        enabled=enabled,
    )


def _input(*, target_rows: int = 3) -> GenerationInput:
    return GenerationInput(
        job_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        tenant_id=str(uuid.uuid4()),
        target_rows=target_rows,
        seed=1,
        modality="video",
    )


def _spec(inp: GenerationInput) -> DatasetSpec:
    return DatasetSpec(
        id=uuid.UUID(inp.dataset_id),
        tenant_id=uuid.UUID(inp.tenant_id),
        name="d",
        description="clips of cats surfing",
        modality=Modality.VIDEO,
        source="description",
        fields=[],
        target_rows=inp.target_rows,
    )


def _configure(repo: _FakeDatasetRepository, registry: VideoProviderRegistry, providers: dict[str, VideoProvider], s3_client: Any) -> None:
    configure_activities(
        ActivityContext(
            repo=repo,
            s3_bucket=_BUCKET,
            s3_client=s3_client,
            video_registry=registry,
            video_providers=providers,
        )
    )


async def test_plan_shards_sizes_video_items_smaller_than_tabular(s3_client: Any) -> None:
    inp = _input(target_rows=10)
    repo = _FakeDatasetRepository()
    repo.specs[uuid.UUID(inp.dataset_id)] = _spec(inp)
    _configure(repo, _FakeVideoProviderRegistry([]), {}, None)

    shards = await plan_shards(inp)

    # Video shards batch only a few heavy clips each -- far smaller than the
    # 50k-row tabular shard -- so 10 items span multiple shards.
    assert len(shards) > 1
    expected_start = 0
    for start, count in shards:
        assert start == expected_start
        expected_start += count
    assert expected_start == inp.target_rows


async def test_generate_shards_uploads_clips_and_manifest_fragment(s3_client: Any) -> None:
    inp = _input(target_rows=2)
    repo = _FakeDatasetRepository()
    repo.specs[uuid.UUID(inp.dataset_id)] = _spec(inp)
    config = _config(tenant_id=uuid.UUID(inp.tenant_id))
    _configure(repo, _FakeVideoProviderRegistry([config]), {"fake": _FakeVideoProvider()}, s3_client)

    keys = await generate_shards(inp, [[0, 2]])

    assert len(keys) == 1
    fragment = json.loads(
        s3_client.get_object(Bucket=_BUCKET, Key=f"{inp.tenant_id}/{keys[0]}")["Body"].read()
    )
    assert [i["index"] for i in fragment] == [0, 1]
    for i in fragment:
        key = i["object_key"]
        assert key == f"datasets/{inp.dataset_id}/{inp.job_id}/videos/item-{i['index']}.mp4"
        stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{inp.tenant_id}/{key}")
        assert stored["Body"].read() == f"clip-{i['seed']}".encode()


async def test_generate_shards_raises_when_no_enabled_provider(s3_client: Any) -> None:
    inp = _input(target_rows=1)
    repo = _FakeDatasetRepository()
    repo.specs[uuid.UUID(inp.dataset_id)] = _spec(inp)
    _configure(repo, _FakeVideoProviderRegistry([_config(enabled=False)]), {}, None)

    with pytest.raises(ValueError, match="no enabled video provider"):
        await generate_shards(inp, [[0, 1]])


async def test_assemble_and_upload_uploads_video_manifest(s3_client: Any) -> None:
    inp = _input(target_rows=1)
    repo = _FakeDatasetRepository()
    repo.specs[uuid.UUID(inp.dataset_id)] = _spec(inp)
    _configure(repo, _FakeVideoProviderRegistry([]), {}, s3_client)
    fragment_key = f"datasets/{inp.dataset_id}/{inp.job_id}/videos/manifest-shard-0.json"
    items = [
        {
            "index": 0,
            "prompt": "x",
            "duration_seconds": 4.0,
            "width": 576,
            "height": 320,
            "fps": 8,
            "seed": 1,
            "provider": "fake",
            "model": "fake-model",
            "content_type": "video/mp4",
            "byte_size": 4,
            "object_key": f"datasets/{inp.dataset_id}/{inp.job_id}/videos/item-0.mp4",
        }
    ]
    s3_client.put_object(
        Bucket=_BUCKET, Key=f"{inp.tenant_id}/{fragment_key}", Body=json.dumps(items).encode()
    )

    key = await assemble_and_upload(inp, [fragment_key])

    expected_key = f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"
    assert key == expected_key
    stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{inp.tenant_id}/{expected_key}")
    manifest = json.loads(stored["Body"].read())
    assert manifest["items"][0]["index"] == 0
    assert manifest["dataset_id"] == inp.dataset_id


async def test_register_version_persists_video_manifest_format() -> None:
    inp = _input(target_rows=1)
    repo = _FakeDatasetRepository()
    repo.specs[uuid.UUID(inp.dataset_id)] = _spec(inp)
    _configure(repo, _FakeVideoProviderRegistry([]), {}, None)

    manifest_key = f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"
    await register_version(inp, manifest_key, rows=1)

    assert repo.versions[0].format == "video-manifest"
