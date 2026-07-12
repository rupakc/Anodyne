from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, ExportArtifact, GenerationJob
from anodyne_dataset.ports import DatasetRepository, Exporter, ExportRepository
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


class _FakeDatasetRepository(DatasetRepository):
    def __init__(self) -> None:
        self.specs: dict[UUID, DatasetSpec] = {}
        self.jobs: dict[UUID, GenerationJob] = {}
        self.versions: dict[UUID, list[DatasetVersion]] = {}

    async def create_spec(self, spec: DatasetSpec) -> None:
        self.specs[spec.id] = spec

    async def get_spec(self, tenant_id: UUID, dataset_id: UUID) -> DatasetSpec | None:
        spec = self.specs.get(dataset_id)
        return spec if spec is not None and spec.tenant_id == tenant_id else None

    async def list_specs(self, tenant_id: UUID) -> list[DatasetSpec]:
        return [s for s in self.specs.values() if s.tenant_id == tenant_id]

    async def update_spec(self, spec: DatasetSpec) -> None:
        self.specs[spec.id] = spec

    async def save_job(self, job: GenerationJob) -> None:
        self.jobs[job.id] = job

    async def get_job(self, tenant_id: UUID, job_id: UUID) -> GenerationJob | None:
        job = self.jobs.get(job_id)
        return job if job is not None and job.tenant_id == tenant_id else None

    async def add_version(self, version: DatasetVersion) -> None:
        self.versions.setdefault(version.dataset_id, []).append(version)

    async def list_versions(self, tenant_id: UUID, dataset_id: UUID) -> list[DatasetVersion]:
        return [v for v in self.versions.get(dataset_id, []) if v.tenant_id == tenant_id]

    async def get_version(self, tenant_id: UUID, version_id: UUID) -> DatasetVersion | None:
        for versions in self.versions.values():
            for v in versions:
                if v.id == version_id and v.tenant_id == tenant_id:
                    return v
        return None


class _FakeExportRepository(ExportRepository):
    def __init__(self) -> None:
        self.exports: list[ExportArtifact] = []

    async def add_export(self, artifact: ExportArtifact) -> None:
        self.exports.append(artifact)

    async def list_exports(self, tenant_id: UUID, dataset_id: UUID) -> list[ExportArtifact]:
        return [a for a in self.exports if a.tenant_id == tenant_id and a.dataset_id == dataset_id]


class _FakeExporter(Exporter):
    """Records the `format` it was called with instead of doing real I/O."""

    def __init__(self) -> None:
        self.calls: list[str | None] = []

    async def export(
        self,
        version: DatasetVersion,
        store: ObjectStore,
        *,
        format: str | None = None,
        batch_size: int = 50_000,
    ) -> ExportArtifact:
        self.calls.append(format)
        resolved = format or ("parquet" if version.row_count > 500_000 else "csv")
        return ExportArtifact(
            id=uuid4(),
            tenant_id=version.tenant_id,
            dataset_id=version.dataset_id,
            version_id=version.id,
            format=resolved,
            row_count=version.row_count,
            object_key=f"datasets/{version.dataset_id}/{version.id}/export.{resolved}",
        )


class _FakeObjectStore(ObjectStore):
    async def put(self, key: str, data: bytes) -> None:
        pass

    async def get(self, key: str) -> bytes:
        return b""

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return []


_Wired = tuple[AsyncClient, Any, _FakeDatasetRepository, _FakeExportRepository, _FakeExporter]


@pytest.fixture
def wired() -> _Wired:
    app = create_app()
    dataset_repo = _FakeDatasetRepository()
    export_repo = _FakeExportRepository()
    exporter = _FakeExporter()
    app.dependency_overrides[deps.get_dataset_repo] = lambda: dataset_repo
    app.dependency_overrides[deps.get_export_repo] = lambda: export_repo
    app.dependency_overrides[deps.get_exporter] = lambda: exporter
    app.dependency_overrides[deps.get_object_store] = lambda: _FakeObjectStore()
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")
    return client, app, dataset_repo, export_repo, exporter


async def test_export_requires_auth(wired) -> None:  # type: ignore[no-untyped-def]
    client, _app, _repo, _export_repo, _exporter = wired
    r = await client.post(f"/datasets/{uuid4()}/versions/{uuid4()}/export", json={})
    assert r.status_code == 401


async def test_export_unknown_version_is_404(wired) -> None:  # type: ignore[no-untyped-def]
    client, app, _repo, _export_repo, _exporter = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(f"/datasets/{uuid4()}/versions/{uuid4()}/export", json={})

    assert r.status_code == 404


async def test_export_rejects_another_tenants_version(wired) -> None:  # type: ignore[no-untyped-def]
    client, app, repo, _export_repo, _exporter = wired
    owner_tid, other_tid = uuid4(), uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=owner_tid,
            dataset_id=dataset_id,
            artifact_uri="k/artifact.parquet",
            row_count=10,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, other_tid)

    r = await client.post(f"/datasets/{dataset_id}/versions/{version_id}/export", json={})

    assert r.status_code == 404


async def test_export_defaults_to_csv_for_small_dataset(wired) -> None:  # type: ignore[no-untyped-def]
    client, app, repo, export_repo, exporter = wired
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="k/artifact.parquet",
            row_count=10,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(f"/datasets/{dataset_id}/versions/{version_id}/export", json={})

    assert r.status_code == 200
    body = r.json()
    assert body["artifact"]["format"] == "csv"
    assert body["url"] == f"https://example.test/{body['artifact']['object_key']}"
    assert exporter.calls == [None]  # no explicit format requested
    assert len(export_repo.exports) == 1


async def test_export_honors_explicit_format(wired) -> None:  # type: ignore[no-untyped-def]
    client, app, repo, _export_repo, exporter = wired
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="k/artifact.parquet",
            row_count=10,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/export", json={"format": "parquet"}
    )

    assert r.status_code == 200
    assert r.json()["artifact"]["format"] == "parquet"
    assert exporter.calls == ["parquet"]


async def test_export_unsupported_format_is_400(wired) -> None:  # type: ignore[no-untyped-def]
    # A well-formed but unsupported format is a client error (400), not a 500 from the exporter.
    client, app, repo, _export_repo, exporter = wired
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="k/artifact.parquet",
            row_count=10,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/export", json={"format": "xml"}
    )

    assert r.status_code == 400
    assert exporter.calls == []  # rejected before the exporter ran


async def test_export_object_is_written_under_tenant_prefix() -> None:
    # End-to-end tenant scoping: with the real per-tenant S3ObjectStore + real PyArrowExporter,
    # the exported object lands under `{tenant_id}/...` and the presigned URL is tenant-scoped --
    # mirroring the generation `assemble_and_upload` tenant-relative-key expectation.
    import io

    import boto3  # type: ignore[import-untyped]
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
    from anodyne_export.exporter import PyArrowExporter
    from anodyne_storage.objectstore import S3ObjectStore
    from moto import mock_aws

    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    bucket = "anodyne-test"
    with mock_aws():
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket=bucket)
        store = S3ObjectStore(bucket, tid, client=s3)

        buf = io.BytesIO()
        pq.write_table(pa.table({"id": [1, 2, 3]}), buf)
        source_key = f"datasets/{dataset_id}/{version_id}/artifact.parquet"
        # Object physically lives under the tenant prefix, exactly as generation stores it.
        s3.put_object(Bucket=bucket, Key=f"{tid}/{source_key}", Body=buf.getvalue())

        version = DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri=source_key,
            format="parquet",
            row_count=3,
        )
        artifact = await PyArrowExporter().export(version, store, format="csv")

        # Returned key is tenant-relative (does not repeat the tenant id) ...
        assert not artifact.object_key.startswith(str(tid))
        assert artifact.object_key == f"datasets/{dataset_id}/{version_id}/export.csv"
        # ... but the object physically landed under the tenant's prefix.
        stored = s3.get_object(Bucket=bucket, Key=f"{tid}/{artifact.object_key}")
        assert stored["Body"].read().startswith(b'"id"')
        # And the presigned URL is scoped to the tenant prefix.
        url = await store.presigned_url(artifact.object_key)
        assert f"{tid}/{artifact.object_key}" in url


async def test_export_is_datasets_read_gated_viewer_allowed(wired) -> None:  # type: ignore[no-untyped-def]
    # Export is a derived-read of data the tenant already owns, exactly like the existing
    # GET .../download route: no write permission required.
    client, app, repo, _export_repo, _exporter = wired
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="k/artifact.parquet",
            row_count=10,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, tid)

    r = await client.post(f"/datasets/{dataset_id}/versions/{version_id}/export", json={})

    assert r.status_code == 200
