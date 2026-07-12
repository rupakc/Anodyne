"""Cheap unit tests for `anodyne_workflows.video_activities` using fakes (no
Temporal, no live infra) -- mirrors `test_activities.py`'s style exactly, but
for the video modality path. `assemble_video_manifest`/`generate_video_items`
are exercised against a real `S3ObjectStore` backed by a moto-mocked bucket
(never a real network call).
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
from anodyne_workflows.video_activities import (
    VideoActivityContext,
    assemble_video_manifest,
    configure_video_activities,
    generate_video_items,
    plan_video_items,
    register_video_version,
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


async def test_plan_video_items_covers_target_rows_in_contiguous_chunks() -> None:
    inp = _input(target_rows=10)

    shards = await plan_video_items(inp)

    expected_start = 0
    for start, count in shards:
        assert start == expected_start
        assert count > 0
        expected_start += count
    assert expected_start == inp.target_rows


async def test_generate_video_items_uploads_clips_and_returns_manifest_items(
    s3_client: Any,
) -> None:
    inp = _input(target_rows=2)
    repo = _FakeDatasetRepository()
    repo.specs[uuid.UUID(inp.dataset_id)] = _spec(inp)
    config = _config(tenant_id=uuid.UUID(inp.tenant_id))
    configure_video_activities(
        VideoActivityContext(
            repo=repo,
            s3_bucket=_BUCKET,
            s3_client=s3_client,
            video_registry=_FakeVideoProviderRegistry([config]),
            providers={"fake": _FakeVideoProvider()},
        )
    )

    items = await generate_video_items(inp, [[0, 2]])

    assert [i["index"] for i in items] == [0, 1]
    for i in items:
        key = i["object_key"]
        assert key == f"datasets/{inp.dataset_id}/{inp.job_id}/videos/item-{i['index']}.mp4"
        stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{inp.tenant_id}/{key}")
        assert stored["Body"].read() == f"clip-{i['seed']}".encode()


async def test_generate_video_items_raises_when_no_enabled_provider() -> None:
    inp = _input(target_rows=1)
    repo = _FakeDatasetRepository()
    repo.specs[uuid.UUID(inp.dataset_id)] = _spec(inp)
    configure_video_activities(
        VideoActivityContext(
            repo=repo,
            s3_bucket=_BUCKET,
            s3_client=None,
            video_registry=_FakeVideoProviderRegistry([_config(enabled=False)]),
            providers={},
        )
    )

    with pytest.raises(ValueError, match="no enabled video provider"):
        await generate_video_items(inp, [[0, 1]])


async def test_assemble_video_manifest_uploads_manifest_json(s3_client: Any) -> None:
    inp = _input(target_rows=1)
    configure_video_activities(
        VideoActivityContext(
            repo=_FakeDatasetRepository(),
            s3_bucket=_BUCKET,
            s3_client=s3_client,
            video_registry=_FakeVideoProviderRegistry([]),
            providers={},
        )
    )
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

    key = await assemble_video_manifest(inp, items)

    expected_key = f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"
    assert key == expected_key
    stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{inp.tenant_id}/{expected_key}")
    manifest = json.loads(stored["Body"].read())
    assert manifest["items"][0]["index"] == 0
    assert manifest["dataset_id"] == inp.dataset_id


async def test_register_video_version_persists_video_manifest_format() -> None:
    inp = _input(target_rows=1)
    repo = _FakeDatasetRepository()
    configure_video_activities(
        VideoActivityContext(
            repo=repo,
            s3_bucket=_BUCKET,
            s3_client=None,
            video_registry=_FakeVideoProviderRegistry([]),
            providers={},
        )
    )
    manifest_key = f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"

    await register_video_version(inp, manifest_key, rows=3)

    assert len(repo.versions) == 1
    version = repo.versions[0]
    assert version.format == "video-manifest"
    assert version.artifact_uri == manifest_key
    assert version.row_count == 3


async def test_generate_video_items_raises_clear_error_when_not_configured() -> None:
    import anodyne_workflows.video_activities as va

    va._ctx = None
    inp = _input(target_rows=1)

    with pytest.raises(RuntimeError, match="not configured"):
        await generate_video_items(inp, [[0, 1]])
