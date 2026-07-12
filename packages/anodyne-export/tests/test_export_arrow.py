import io
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetVersion
from anodyne_export.exporter import PyArrowExporter


class _FakeObjectStore(ObjectStore):
    def __init__(self) -> None:
        self.objects: dict[str, bytes] = {}

    async def put(self, key: str, data: bytes) -> None:
        self.objects[key] = data

    async def get(self, key: str) -> bytes:
        return self.objects[key]

    async def presigned_url(self, key: str, expires: int = 3600) -> str:
        return f"https://example.test/{key}"

    async def list(self, prefix: str) -> list[str]:
        return [k for k in self.objects if k.startswith(prefix)]


def _fixture_table() -> pa.Table:
    n = 50
    return pa.table(
        {
            "id": list(range(n)),
            "score": [None if i % 7 == 0 else i * 1.5 for i in range(n)],
            "name": [f"café-{i}" if i % 3 == 0 else f"user-{i}" for i in range(n)],
        }
    )


async def _stored_version(store: _FakeObjectStore, table: pa.Table) -> DatasetVersion:
    buf = io.BytesIO()
    pq.write_table(table, buf)
    key = "datasets/d/v/artifact.parquet"
    await store.put(key, buf.getvalue())
    return DatasetVersion(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        artifact_uri=key,
        format="parquet",
        row_count=table.num_rows,
    )


async def test_arrow_export_round_trips_fixture_table() -> None:
    store = _FakeObjectStore()
    table = _fixture_table()
    version = await _stored_version(store, table)

    artifact = await PyArrowExporter().export(version, store, format="arrow")

    assert artifact.format == "arrow"
    arrow_bytes = await store.get(artifact.object_key)
    with pa.ipc.open_stream(pa.BufferReader(arrow_bytes)) as reader:
        restored = reader.read_all()
    assert restored.equals(table)
