from uuid import UUID, uuid4

import pytest
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetVersion, ExportArtifact
from anodyne_dataset.ports import Exporter, ExportRepository


class _EchoExporter(Exporter):
    async def export(
        self,
        version: DatasetVersion,
        store: ObjectStore,
        *,
        format: str | None = None,
        batch_size: int = 50_000,
    ) -> ExportArtifact:
        return ExportArtifact(
            id=uuid4(),
            tenant_id=version.tenant_id,
            dataset_id=version.dataset_id,
            version_id=version.id,
            format=format or "csv",
            row_count=version.row_count,
            object_key="datasets/d/v/export.csv",
        )


class _NoopObjectStore(ObjectStore):
    async def put(self, key: str, data: bytes) -> None:
        pass

    async def get(self, key: str) -> bytes:
        return b""

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return []


async def test_exporter_is_an_abstract_async_contract() -> None:
    version = DatasetVersion(
        id=uuid4(), tenant_id=uuid4(), dataset_id=uuid4(), artifact_uri="k/artifact.parquet"
    )
    out = await _EchoExporter().export(version, _NoopObjectStore())
    assert out.format == "csv"
    assert out.version_id == version.id

    with pytest.raises(TypeError):
        Exporter()  # type: ignore[abstract]


class _InMemoryExportRepository(ExportRepository):
    def __init__(self) -> None:
        self.exports: list[ExportArtifact] = []

    async def add_export(self, artifact: ExportArtifact) -> None:
        self.exports.append(artifact)

    async def list_exports(self, tenant_id: UUID, dataset_id: UUID) -> list[ExportArtifact]:
        return [a for a in self.exports if a.tenant_id == tenant_id and a.dataset_id == dataset_id]


async def test_export_repository_is_an_abstract_async_contract() -> None:
    repo = _InMemoryExportRepository()
    tid, did = uuid4(), uuid4()
    artifact = ExportArtifact(
        id=uuid4(),
        tenant_id=tid,
        dataset_id=did,
        version_id=uuid4(),
        format="csv",
        row_count=1,
        object_key="datasets/d/v/export.csv",
    )

    await repo.add_export(artifact)

    assert await repo.list_exports(tid, did) == [artifact]
    assert await repo.list_exports(uuid4(), did) == []

    with pytest.raises(TypeError):
        ExportRepository()  # type: ignore[abstract]
