from __future__ import annotations

from uuid import UUID, uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetSpec, DatasetVersion, ExportArtifact, GenerationJob
from anodyne_dataset.ports import DatasetRepository, Exporter, ExportRepository
from anodyne_graph.models import Edge, GraphOntology, Node, compute_metrics
from anodyne_graph.models import GraphDataset as _GraphDataset
from anodyne_graph.serialization import to_json_bytes
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID) -> TenantContext:
    u = User(id=uuid4(), tenant_id=tenant_id, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tenant_id, user=u, roles=[role])


def _graph_dataset() -> _GraphDataset:
    nodes = [Node(id="p1", type="Person"), Node(id="c1", type="Company")]
    edges = [Edge(id="e1", type="WORKS_AT", source="p1", target="c1")]
    return _GraphDataset(
        ontology=GraphOntology(), nodes=nodes, edges=edges, metrics=compute_metrics(nodes, edges)
    )


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


class _RecordingExporter(Exporter):
    """Records every call; used to prove *which* exporter dependency fired."""

    def __init__(self, name: str) -> None:
        self.name = name
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
        resolved = format or "csv"
        return ExportArtifact(
            id=uuid4(),
            tenant_id=version.tenant_id,
            dataset_id=version.dataset_id,
            version_id=version.id,
            format=resolved,
            row_count=version.row_count,
            object_key=f"datasets/{version.dataset_id}/{version.id}/export.{resolved}",
        )


class _GraphObjectStore(ObjectStore):
    """A real (in-memory) keyed store, seeded with the source graph_json bytes under every
    key -- so the real `GraphExporter` can both read the source artifact and have its written
    export bytes read back correctly, unlike a store that ignores the key entirely."""

    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def get(self, key: str) -> bytes:
        return self.objects.get(key, to_json_bytes(_graph_dataset()))

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return []


@pytest.fixture
def app_and_repos():  # type: ignore[no-untyped-def]
    app = create_app()
    dataset_repo = _FakeDatasetRepository()
    export_repo = _FakeExportRepository()
    tabular_exporter = _RecordingExporter("tabular")
    graph_exporter = _RecordingExporter("graph")
    app.dependency_overrides[deps.get_dataset_repo] = lambda: dataset_repo
    app.dependency_overrides[deps.get_export_repo] = lambda: export_repo
    app.dependency_overrides[deps.get_exporter] = lambda: tabular_exporter
    app.dependency_overrides[deps.get_graph_exporter] = lambda: graph_exporter
    app.dependency_overrides[deps.get_object_store] = lambda: _GraphObjectStore()
    return app, dataset_repo, tabular_exporter, graph_exporter


async def test_graph_json_version_dispatches_to_graph_exporter(app_and_repos) -> None:  # type: ignore[no-untyped-def]
    app, repo, tabular_exporter, graph_exporter = app_and_repos
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="datasets/x/y/graph.json",
            format="graph_json",
            row_count=3,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/export", json={"format": "ttl"}
    )

    assert r.status_code == 200
    assert graph_exporter.calls == ["ttl"]
    assert tabular_exporter.calls == []


async def test_tabular_version_still_dispatches_to_pyarrow_exporter(app_and_repos) -> None:  # type: ignore[no-untyped-def]
    app, repo, tabular_exporter, graph_exporter = app_and_repos
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="k/artifact.parquet",
            format="parquet",
            row_count=10,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")

    r = await client.post(f"/datasets/{dataset_id}/versions/{version_id}/export", json={})

    assert r.status_code == 200
    assert tabular_exporter.calls == [None]
    assert graph_exporter.calls == []


async def test_graph_format_is_accepted_by_supported_formats_validation(app_and_repos) -> None:  # type: ignore[no-untyped-def]
    # A graph-only format string (never valid for PyArrowExporter) must not be rejected as 400
    # up front -- validation now covers the union of tabular + graph formats.
    app, repo, _tabular_exporter, graph_exporter = app_and_repos
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="datasets/x/y/graph.json",
            format="graph_json",
            row_count=3,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/export", json={"format": "neo4j-csv"}
    )

    assert r.status_code == 200
    assert graph_exporter.calls == ["neo4j-csv"]


async def test_tabular_format_against_graph_version_is_400_not_500(app_and_repos) -> None:  # type: ignore[no-untyped-def]
    # `format=csv` is in the merged SUPPORTED_FORMATS gate (it's valid for the tabular exporter),
    # but a graph_json version dispatches to GraphExporter, which doesn't support "csv" -- that
    # must surface as a 400, not an unhandled 500.
    from anodyne_graph.export import GraphExporter

    app, repo, _tabular_exporter, _graph_exporter = app_and_repos
    app.dependency_overrides[deps.get_graph_exporter] = lambda: GraphExporter()
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="datasets/x/y/graph.json",
            format="graph_json",
            row_count=3,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/export", json={"format": "csv"}
    )

    assert r.status_code == 400


async def test_end_to_end_real_graph_exporter_produces_graphml(app_and_repos) -> None:  # type: ignore[no-untyped-def]
    # Swap in the *real* GraphExporter to prove the whole path (route -> GraphExporter ->
    # rdflib/networkx encoders -> streamed response) works, not just the fake dispatch.
    from anodyne_graph.export import GraphExporter

    app, repo, _tabular_exporter, _graph_exporter = app_and_repos
    app.dependency_overrides[deps.get_graph_exporter] = lambda: GraphExporter()
    tid = uuid4()
    dataset_id, version_id = uuid4(), uuid4()
    await repo.add_version(
        DatasetVersion(
            id=version_id,
            tenant_id=tid,
            dataset_id=dataset_id,
            artifact_uri="datasets/x/y/graph.json",
            format="graph_json",
            row_count=3,
        )
    )
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://t")

    r = await client.post(
        f"/datasets/{dataset_id}/versions/{version_id}/export", json={"format": "graphml"}
    )

    assert r.status_code == 200
    assert r.headers["content-type"] == "application/xml"
    assert r.headers["content-disposition"].endswith('.graphml"')
    assert b"<graphml" in r.content
