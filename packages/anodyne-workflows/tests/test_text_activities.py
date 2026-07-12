"""Modality-aware branching in `anodyne_workflows.activities` for text datasets.

Regression discipline: these tests use their own local fakes (not the shared
`_FakeDatasetRepository` in `test_activities.py`, whose `get_spec` always
returns `None`) so the modality branches are actually exercised. The existing
`test_activities.py` file/assertions are untouched.
"""

from __future__ import annotations

import io
import json
import uuid
from collections.abc import Generator as PyGenerator
from typing import Any

import anodyne_compute
import boto3  # type: ignore[import-untyped]
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
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

_BUCKET = "test-bucket"


@pytest.fixture
def s3_client() -> PyGenerator[Any, None, None]:
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket=_BUCKET)
        yield c


class _FakeModelRegistry:
    def __init__(self, configs: dict[uuid.UUID, ModelConfig]) -> None:
        self._configs = configs

    async def get(self, tenant_id: uuid.UUID, config_id: uuid.UUID) -> ModelConfig | None:
        cfg = self._configs.get(config_id)
        return cfg if cfg is not None and cfg.tenant_id == tenant_id else None


class _FakeDatasetRepository(DatasetRepository):
    """Unlike `test_activities.py`'s fake, `get_spec` returns a real spec."""

    def __init__(self, spec: DatasetSpec) -> None:
        self._spec = spec
        self.versions: list[DatasetVersion] = []

    async def create_spec(self, spec: DatasetSpec) -> None: ...

    async def get_spec(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> DatasetSpec | None:
        if self._spec.tenant_id == tenant_id and self._spec.id == dataset_id:
            return self._spec
        return None

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


def _text_spec(tenant_id: uuid.UUID, dataset_id: uuid.UUID, target_rows: int = 10) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,
        tenant_id=tenant_id,
        name="d",
        description="support tickets",
        modality=Modality.TEXT,
        source="description",
        fields=[
            FieldSpec(name="text", semantic_type=SemanticType.TEXT),
            FieldSpec(name="label", semantic_type=SemanticType.CATEGORICAL),
        ],
        target_rows=target_rows,
    )


def _tabular_spec(
    tenant_id: uuid.UUID, dataset_id: uuid.UUID, target_rows: int = 10
) -> DatasetSpec:
    return DatasetSpec(
        id=dataset_id,
        tenant_id=tenant_id,
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="description",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=target_rows,
    )


def _input(
    tenant_id: uuid.UUID,
    dataset_id: uuid.UUID,
    target_rows: int,
    model_config_id: str | None = None,
) -> GenerationInput:
    return GenerationInput(
        job_id=str(uuid.uuid4()),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=target_rows,
        seed=1,
        model_config_id=model_config_id,
    )


async def test_plan_shards_sizes_text_shards_smaller_than_tabular() -> None:
    tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    spec = _text_spec(tenant_id, dataset_id, target_rows=450)
    configure_activities(
        ActivityContext(repo=_FakeDatasetRepository(spec), s3_bucket=_BUCKET, s3_client=None)
    )

    shards = await plan_shards(_input(tenant_id, dataset_id, 450))

    assert all(count <= 200 for _start, count in shards)
    assert sum(count for _, count in shards) == 450


async def test_plan_shards_keeps_tabular_shard_size_unchanged() -> None:
    tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    spec = _tabular_spec(tenant_id, dataset_id, target_rows=125_000)
    configure_activities(
        ActivityContext(repo=_FakeDatasetRepository(spec), s3_bucket=_BUCKET, s3_client=None)
    )

    shards = await plan_shards(_input(tenant_id, dataset_id, 125_000))

    assert shards[0] == [0, 50_000]
    assert sum(count for _, count in shards) == 125_000


async def test_generate_shards_dispatches_text_path(monkeypatch: pytest.MonkeyPatch) -> None:
    tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    spec = _text_spec(tenant_id, dataset_id, target_rows=5)
    model_config = ModelConfig(
        id=uuid.uuid4(), tenant_id=tenant_id, name="m", provider="ollama", model="llama3"
    )
    registry = _FakeModelRegistry({model_config.id: model_config})
    configure_activities(
        ActivityContext(
            repo=_FakeDatasetRepository(spec),
            s3_bucket=_BUCKET,
            s3_client=None,
            model_registry=registry,
            secret_key="k",
        )
    )

    calls: list[tuple[Any, ...]] = []

    class _FakeRemoteResult:
        def __init__(self, value: bytes) -> None:
            self._value = value

    def _fake_remote(*args: Any) -> _FakeRemoteResult:
        calls.append(args)
        table = pa.table({"text": ["a"], "label": ["b"]})
        buf = io.BytesIO()
        pq.write_table(table, buf)
        return _FakeRemoteResult(buf.getvalue())

    monkeypatch.setattr(
        anodyne_compute.remote_generate_text_shard, "remote", _fake_remote, raising=False
    )
    monkeypatch.setattr(
        "anodyne_workflows.activities.ray.get",
        lambda ref: ref._value,
    )
    monkeypatch.setattr("anodyne_workflows.activities._object_store", lambda inp: _FakeStore())

    keys = await generate_shards(_input(tenant_id, dataset_id, 5, str(model_config.id)), [[0, 5]])

    assert len(keys) == 1
    assert len(calls) == 1
    assert calls[0][0] is spec
    assert calls[0][1] is model_config
    assert calls[0][2] == "k"


class _FakeStore:
    def __init__(self) -> None:
        self.put_calls: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.put_calls[key] = data

    async def get(self, key: str) -> bytes:
        return self.put_calls[key]


async def test_generate_shards_raises_without_model_config_for_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    spec = _text_spec(tenant_id, dataset_id, target_rows=5)
    configure_activities(
        ActivityContext(
            repo=_FakeDatasetRepository(spec),
            s3_bucket=_BUCKET,
            s3_client=None,
            model_registry=_FakeModelRegistry({}),
            secret_key="k",
        )
    )

    with pytest.raises(ValueError, match="model"):
        await generate_shards(_input(tenant_id, dataset_id, 5, model_config_id=None), [[0, 5]])


async def test_assemble_and_upload_writes_jsonl_and_manifest_for_text(s3_client: Any) -> None:
    tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    spec = _text_spec(tenant_id, dataset_id, target_rows=2)
    inp = _input(tenant_id, dataset_id, 2, model_config_id=str(uuid.uuid4()))
    configure_activities(
        ActivityContext(repo=_FakeDatasetRepository(spec), s3_bucket=_BUCKET, s3_client=s3_client)
    )
    table = pa.table({"text": ["a support row", "another row"], "label": ["x", "y"]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    shard_key = f"datasets/{inp.dataset_id}/{inp.job_id}/shard-0.parquet"
    s3_client.put_object(Bucket=_BUCKET, Key=f"{tenant_id}/{shard_key}", Body=buf.getvalue())

    artifact_key = await assemble_and_upload(inp, [shard_key])

    assert artifact_key == f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.jsonl"
    jsonl_bytes = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{artifact_key}")[
        "Body"
    ].read()
    lines = jsonl_bytes.decode().strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["text"] == "a support row"

    manifest_key = f"datasets/{inp.dataset_id}/{inp.job_id}/manifest.json"
    manifest = json.loads(
        s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{manifest_key}")["Body"].read()
    )
    assert manifest["modality"] == "text"
    assert set(manifest["fields"]) == {"text", "label"}
    assert manifest["rows_produced"] == 2


async def test_register_version_sets_jsonl_format_for_text_artifact() -> None:
    tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    spec = _text_spec(tenant_id, dataset_id)
    repo = _FakeDatasetRepository(spec)
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))
    inp = _input(tenant_id, dataset_id, 10)
    artifact_key = f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.jsonl"

    await register_version(inp, artifact_key, rows=10)

    assert repo.versions[0].format == "jsonl"


async def test_register_version_keeps_parquet_format_for_tabular_artifact() -> None:
    tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4()
    spec = _tabular_spec(tenant_id, dataset_id)
    repo = _FakeDatasetRepository(spec)
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))
    inp = _input(tenant_id, dataset_id, 10)
    artifact_key = f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.parquet"

    await register_version(inp, artifact_key, rows=10)

    assert repo.versions[0].format == "parquet"
