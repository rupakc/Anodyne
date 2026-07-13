"""Cheap unit tests for `anodyne_workflows.activities` using fakes (no Temporal, no infra).

`plan_shards` is exercised as a plain async function (it's pure). `set_status`
is exercised via the context-injection mechanism (`configure_activities`) with
a fake `DatasetRepository` (no object-store access needed). `assemble_and_upload`
is exercised against a real `S3ObjectStore` backed by a moto-mocked S3 bucket
-- not a fake -- specifically to catch tenant-prefix bugs like the one this
suite regression-tests (BLOCKER 2: worker wrote under a nil-namespace prefix
while the gateway presigned under the real tenant prefix, so keys never
matched. See `anodyne_workflows.activities.ActivityContext` docstring).
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
    ColumnProfile,
    DatasetSpec,
    DatasetVersion,
    FieldSpec,
    GenerationJob,
    JobStatus,
    Modality,
    Profile,
    SemanticType,
)
from anodyne_dataset.ports import DatasetRepository, ProfileRepository
from anodyne_workflows import handlers as handlers_module
from anodyne_workflows.activities import (
    ActivityContext,
    assemble_and_upload,
    configure_activities,
    generate_shards,
    plan_shards,
    register_version,
    set_status,
)
from anodyne_workflows.workflow import GenerationInput
from moto import mock_aws

_BUCKET = "test-bucket"


@pytest.fixture
def s3_client() -> Generator[Any, None, None]:
    with mock_aws():
        c = boto3.client("s3", region_name="us-east-1")
        c.create_bucket(Bucket=_BUCKET)
        yield c


class _FakePublisher:
    """Records every published (channel, message) pair for assertion."""

    def __init__(self) -> None:
        self.messages: list[tuple[str, str]] = []

    async def publish(self, channel: str, message: str) -> None:
        self.messages.append((channel, message))


class _FakeDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self.jobs: dict[uuid.UUID, GenerationJob] = {}
        self.versions: list[DatasetVersion] = []
        self.specs: dict[uuid.UUID, DatasetSpec] = {}

    async def create_spec(self, spec: DatasetSpec) -> None:
        self.specs[spec.id] = spec

    async def get_spec(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> DatasetSpec | None:
        return self.specs.get(dataset_id)

    async def list_specs(self, tenant_id: uuid.UUID) -> list[DatasetSpec]:
        return []

    async def update_spec(self, spec: DatasetSpec) -> None: ...

    async def save_job(self, job: GenerationJob) -> None:
        self.jobs[job.id] = job

    async def get_job(self, tenant_id: uuid.UUID, job_id: uuid.UUID) -> GenerationJob | None:
        return self.jobs.get(job_id)

    async def add_version(self, version: DatasetVersion) -> None:
        self.versions.append(version)

    async def list_versions(
        self, tenant_id: uuid.UUID, dataset_id: uuid.UUID
    ) -> list[DatasetVersion]:
        return []

    async def get_version(
        self, tenant_id: uuid.UUID, version_id: uuid.UUID
    ) -> DatasetVersion | None:
        return next((v for v in self.versions if v.id == version_id), None)


def _input(job_id: uuid.UUID, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> GenerationInput:
    return GenerationInput(
        job_id=str(job_id),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=10,
        seed=1,
    )


async def test_set_status_preserves_workflow_id_and_message() -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    repo = _FakeDatasetRepository()
    repo.jobs[job_id] = GenerationJob(
        id=job_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        status=JobStatus.PENDING,
        progress=0.0,
        message="created by gateway",
        workflow_id="wf-x",
    )
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))

    await set_status(_input(job_id, tenant_id, dataset_id), "running", 0.5)

    stored = repo.jobs[job_id]
    assert stored.workflow_id == "wf-x"
    assert stored.message == "created by gateway"  # untouched: no message passed
    assert stored.status == JobStatus.RUNNING
    assert stored.progress == 0.5


async def test_set_status_updates_message_when_passed() -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    repo = _FakeDatasetRepository()
    repo.jobs[job_id] = GenerationJob(
        id=job_id,
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        workflow_id="wf-x",
        message="old",
    )
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))

    await set_status(_input(job_id, tenant_id, dataset_id), "failed", 0.4, message="boom")

    stored = repo.jobs[job_id]
    assert stored.workflow_id == "wf-x"
    assert stored.message == "boom"
    assert stored.status == JobStatus.FAILED


async def test_set_status_publishes_to_per_job_channel() -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    repo = _FakeDatasetRepository()
    repo.jobs[job_id] = GenerationJob(id=job_id, tenant_id=tenant_id, dataset_id=dataset_id)
    publisher = _FakePublisher()
    configure_activities(
        ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None, publisher=publisher)
    )

    await set_status(_input(job_id, tenant_id, dataset_id), "running", 0.5)

    assert len(publisher.messages) == 1
    channel, message = publisher.messages[0]
    # Must match the gateway's per-job WS subscription (`f"job:{job_id}"`),
    # NOT a per-tenant channel -- otherwise live progress never reaches the
    # `/jobs/{job_id}/stream` websocket.
    assert channel == f"job:{job_id}"
    payload = json.loads(message)
    assert payload["status"] == "running"
    assert payload["progress"] == 0.5


async def test_assemble_and_upload_returns_tenant_relative_key(s3_client: Any) -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    inp = _input(job_id, tenant_id, dataset_id)
    table = pa.table({"x": [1, 2, 3]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    # Tenant-relative key, as `generate_shards` would produce it -- physically
    # written under the tenant prefix (mirrors what `S3ObjectStore.put` does).
    shard_key = f"datasets/{inp.dataset_id}/{inp.job_id}/shard-0.parquet"
    s3_client.put_object(Bucket=_BUCKET, Key=f"{tenant_id}/{shard_key}", Body=buf.getvalue())
    configure_activities(
        ActivityContext(repo=_FakeDatasetRepository(), s3_bucket=_BUCKET, s3_client=s3_client)
    )

    artifact_key = await assemble_and_upload(inp, [shard_key])

    expected_key = f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.parquet"
    assert artifact_key == expected_key
    assert not artifact_key.startswith("http://")
    assert not artifact_key.startswith("https://")
    # Regression check for BLOCKER 2: the artifact must be physically stored
    # under exactly one tenant prefix -- `{tenant}/{key}` -- not double-nested
    # under a separate worker-side namespace the gateway's presigner never
    # looks under.
    stored = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{expected_key}")
    assert pq.read_table(io.BytesIO(stored["Body"].read())).num_rows == 3


async def test_register_version_persists_artifact_key_not_url() -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    inp = _input(job_id, tenant_id, dataset_id)
    repo = _FakeDatasetRepository()
    configure_activities(ActivityContext(repo=repo, s3_bucket=_BUCKET, s3_client=None))
    artifact_key = f"datasets/{inp.dataset_id}/{inp.job_id}/artifact.parquet"

    await register_version(inp, artifact_key, rows=10)

    assert len(repo.versions) == 1
    stored = repo.versions[0]
    assert stored.artifact_uri == artifact_key
    assert not stored.artifact_uri.startswith("http")
    # Tenant-relative: the gateway's per-tenant `S3ObjectStore` will prepend
    # `{tenant_id}/` itself when presigning -- this must not repeat it.
    assert not stored.artifact_uri.startswith(str(tenant_id))


async def test_plan_shards_covers_target_rows_in_contiguous_chunks() -> None:
    inp = GenerationInput(
        job_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        tenant_id=str(uuid.uuid4()),
        target_rows=125_000,
        seed=1,
    )

    shards = await plan_shards(inp)

    assert shards[0][0] == 0
    expected_start = 0
    for start, count in shards:
        assert start == expected_start
        assert count > 0
        expected_start += count
    assert expected_start == inp.target_rows
    assert sum(count for _, count in shards) == inp.target_rows


def test_generation_input_still_constructs_without_method_kwarg() -> None:
    # Regression: every C0 call site constructs `GenerationInput` with only
    # job_id/dataset_id/tenant_id/target_rows/seed -- `method` must default.
    inp = GenerationInput(
        job_id=str(uuid.uuid4()),
        dataset_id=str(uuid.uuid4()),
        tenant_id=str(uuid.uuid4()),
        target_rows=10,
        seed=1,
    )
    assert inp.method == "copula"


class _FakeProfileRepository(ProfileRepository):
    def __init__(self) -> None:
        self.profiles: dict[uuid.UUID, Profile] = {}

    async def save_profile(self, profile: Profile) -> None:
        self.profiles[profile.dataset_id] = profile

    async def get_profile(self, tenant_id: uuid.UUID, dataset_id: uuid.UUID) -> Profile | None:
        p = self.profiles.get(dataset_id)
        return p if p is not None and p.tenant_id == tenant_id else None


class _FakeGenerator:
    """Deliberately has no side-channel state: `generate_shards` dispatches shard sampling
    to real Ray remote tasks, which run in a *different process* -- mutations to a shared
    Python list captured by a `generate()` closure would silently vanish across that
    boundary. Correctness is instead verified from `generate_shards`' actual return value
    (the uploaded per-shard artifacts), not from call-count side effects.
    """

    def generate(self, spec: DatasetSpec, start_row: int, count: int, seed: int) -> pa.Table:
        return pa.table({"age": [start_row + i for i in range(count)]})


async def test_generate_shards_from_sample_fits_once_and_samples_per_shard(
    s3_client: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    inp = GenerationInput(
        job_id=str(job_id),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=20,
        seed=1,
        method="copula",
    )
    spec = DatasetSpec(
        id=dataset_id,
        tenant_id=tenant_id,
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="sample",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=20,
    )
    repo = _FakeDatasetRepository()
    repo.specs[dataset_id] = spec
    profile_repo = _FakeProfileRepository()
    profile = Profile(
        id=uuid.uuid4(),
        tenant_id=tenant_id,
        dataset_id=dataset_id,
        row_count=3,
        columns=[ColumnProfile(name="age", semantic_type=SemanticType.INTEGER, min=0.0, max=9.0)],
        sample_uri="datasets/x/sample/data.csv",
        sample_filename="data.csv",
    )
    profile_repo.profiles[dataset_id] = profile
    s3_client.put_object(
        Bucket=_BUCKET,
        Key=f"{tenant_id}/datasets/x/sample/data.csv",
        Body=b"age\n1\n2\n3\n",
    )
    configure_activities(
        ActivityContext(
            repo=repo, s3_bucket=_BUCKET, s3_client=s3_client, profile_repo=profile_repo
        )
    )
    build_calls: list[str] = []

    def _fake_build(method: str, prof: Profile, df: Any, **kwargs: Any) -> _FakeGenerator:
        build_calls.append(method)
        return _FakeGenerator()

    monkeypatch.setattr(handlers_module, "build_tabular_generator", _fake_build)

    keys = await generate_shards(inp, [[0, 10], [10, 10]])

    assert build_calls == ["copula"]  # fit once, not once per shard
    assert len(keys) == 2
    # Each shard was sampled independently (disjoint start_row) by the *same*
    # fitted generator -- verified via the actual uploaded artifacts, not a
    # cross-process side channel (see `_FakeGenerator`'s docstring).
    row_counts = []
    for key in keys:
        obj = s3_client.get_object(Bucket=_BUCKET, Key=f"{tenant_id}/{key}")
        table = pq.read_table(io.BytesIO(obj["Body"].read()))
        row_counts.append(table.num_rows)
        assert table.column("age").to_pylist()[0] in (0, 10)  # matches its shard's start_row
    assert row_counts == [10, 10]


async def test_generate_shards_from_sample_without_profile_raises() -> None:
    job_id, tenant_id, dataset_id = uuid.uuid4(), uuid.uuid4(), uuid.uuid4()
    inp = GenerationInput(
        job_id=str(job_id),
        dataset_id=str(dataset_id),
        tenant_id=str(tenant_id),
        target_rows=20,
        seed=1,
    )
    spec = DatasetSpec(
        id=dataset_id,
        tenant_id=tenant_id,
        name="d",
        description="",
        modality=Modality.TABULAR,
        source="sample",
        fields=[FieldSpec(name="age", semantic_type=SemanticType.INTEGER)],
        target_rows=20,
    )
    repo = _FakeDatasetRepository()
    repo.specs[dataset_id] = spec
    configure_activities(
        ActivityContext(
            repo=repo, s3_bucket=_BUCKET, s3_client=None, profile_repo=_FakeProfileRepository()
        )
    )

    with pytest.raises(ValueError, match="no profile"):
        await generate_shards(inp, [[0, 10]])
