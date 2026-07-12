"""Tests for the `spec.modality`-keyed audio dispatch in `anodyne_workflows.activities`.

Mirrors `test_activities.py`'s fake-repo + moto-S3 style. Uses a mocked
`AudioProvider` throughout -- no real inference, no network -- per the C4
CRITICAL constraint (no GPU/provider keys in this environment).
"""

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Generator
from typing import Any

import boto3  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
from anodyne_dataset.models import (
    AudioSynthesisRequest,
    AudioSynthesisResult,
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    GenerationJob,
    Modality,
    SemanticType,
)
from anodyne_dataset.ports import AudioProvider, DatasetRepository
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


class _MockProvider(AudioProvider):
    async def synthesize(self, request: AudioSynthesisRequest) -> AudioSynthesisResult:
        return AudioSynthesisResult(audio_bytes=request.text.encode(), format="wav")


async def _mock_provider_factory(spec: DatasetSpec) -> AudioProvider:
    return _MockProvider()


class _FakeRepo(DatasetRepository):
    def __init__(self, spec: DatasetSpec | None) -> None:
        self._spec = spec
        self.versions: list[DatasetVersion] = []

    async def create_spec(self, spec: DatasetSpec) -> None: ...

    async def get_spec(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> DatasetSpec | None:
        return self._spec

    async def list_specs(self, tenant_id: uuid.UUID) -> list[DatasetSpec]:
        return []

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


@pytest.fixture
def s3_client() -> Generator[Any, None, None]:
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket=_BUCKET)
        yield c


def _audio_spec(tenant_id: uuid.UUID, dataset_id: uuid.UUID, rows: int = 4) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,
        tenant_id=tenant_id,
        name="d",
        description="",
        modality=Modality.AUDIO,
        source="description",
        fields=[FieldSpec(name="transcript", semantic_type=SemanticType.TEXT)],
        target_rows=rows,
        directives={"audio": {"prompts": [f"t{i}" for i in range(rows)]}},
    )


def _input(
    job_id: uuid.UUID, tenant_id: uuid.UUID, dataset_id: uuid.UUID, rows: int
) -> GenerationInput:
    return GenerationInput(
        job_id=str(job_id),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=rows,
        seed=1,
    )


async def test_generate_shards_uploads_items_and_manifest_fragment(s3_client: Any) -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _audio_spec(tenant_id, dataset_id)
    repo = _FakeRepo(spec)
    configure_activities(
        ActivityContext(
            repo=repo,
            s3_bucket=_BUCKET,
            s3_client=s3_client,
            audio_provider_factory=_mock_provider_factory,
        )
    )
    inp = _input(job_id, tenant_id, dataset_id, 4)

    keys = await generate_shards(inp, [[0, 4]])

    assert len(keys) == 1
    fragment = json.loads(
        s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{keys[0]}")["Body"].read()
    )
    assert [item["text"] for item in fragment] == ["t0", "t1", "t2", "t3"]
    for item in fragment:
        stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{item['object_key']}")
        assert stored["Body"].read() == item["text"].encode()


async def test_generate_shards_raises_without_audio_provider_factory(s3_client: Any) -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _audio_spec(tenant_id, dataset_id)
    configure_activities(
        ActivityContext(repo=_FakeRepo(spec), s3_bucket=_BUCKET, s3_client=s3_client)
    )
    inp = _input(job_id, tenant_id, dataset_id, 4)

    with pytest.raises(RuntimeError, match="audio_provider_factory"):
        await generate_shards(inp, [[0, 4]])


async def test_generate_shards_tabular_path_is_unaffected(s3_client: Any) -> None:
    # Regression guard: a `None` spec (as every pre-existing tabular fake
    # repo returns) must still take the original tabular Ray-shard path.
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    configure_activities(
        ActivityContext(repo=_FakeRepo(None), s3_bucket=_BUCKET, s3_client=s3_client)
    )
    inp = _input(job_id, tenant_id, dataset_id, 4)

    with pytest.raises(ValueError, match="not found"):
        await generate_shards(inp, [[0, 4]])


async def test_assemble_and_upload_merges_audio_manifest_fragments(s3_client: Any) -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _audio_spec(tenant_id, dataset_id, rows=2)
    inp = _input(job_id, tenant_id, dataset_id, 2)
    fragment_key = f"datasets/{dataset_id}/{job_id}/audio/manifest-shard-0.json"
    fragment = [
        {
            "index": 0,
            "object_key": "x",
            "text": "t0",
            "label": None,
            "voice": None,
            "format": "wav",
            "duration_seconds": None,
        }
    ]
    s3_client.put_object(
        Bucket=_BUCKET, Key=f"{tenant_id}/{fragment_key}", Body=json.dumps(fragment).encode()
    )
    configure_activities(
        ActivityContext(repo=_FakeRepo(spec), s3_bucket=_BUCKET, s3_client=s3_client)
    )

    artifact_key = await assemble_and_upload(inp, [fragment_key])

    assert artifact_key == f"datasets/{dataset_id}/{job_id}/manifest.json"
    manifest = json.loads(
        s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{artifact_key}")["Body"].read()
    )
    assert manifest["items"][0]["text"] == "t0"


async def test_assemble_and_upload_tabular_path_is_unaffected(s3_client: Any) -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    inp = _input(job_id, tenant_id, dataset_id, 3)
    table = pa.table({"x": [1, 2, 3]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    shard_key = f"datasets/{dataset_id}/{job_id}/shard-0.parquet"
    s3_client.put_object(Bucket=_BUCKET, Key=f"{tenant_id}/{shard_key}", Body=buf.getvalue())
    configure_activities(
        ActivityContext(repo=_FakeRepo(None), s3_bucket=_BUCKET, s3_client=s3_client)
    )

    artifact_key = await assemble_and_upload(inp, [shard_key])

    expected_key = f"datasets/{dataset_id}/{job_id}/artifact.parquet"
    assert artifact_key == expected_key
    stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{expected_key}")
    assert pq.read_table(io.BytesIO(stored["Body"].read())).num_rows == 3


async def test_register_version_sets_audio_manifest_format() -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    spec = _audio_spec(tenant_id, dataset_id)
    repo = _FakeRepo(spec)
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))
    inp = _input(job_id, tenant_id, dataset_id, 4)

    await register_version(inp, "datasets/x/manifest.json", rows=4)

    assert repo.versions[0].format == "audio_manifest"


async def test_register_version_tabular_path_defaults_to_parquet() -> None:
    tenant_id, dataset_id, job_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    repo = _FakeRepo(None)
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))
    inp = _input(job_id, tenant_id, dataset_id, 10)

    await register_version(inp, "datasets/x/artifact.parquet", rows=10)

    assert repo.versions[0].format == "parquet"
