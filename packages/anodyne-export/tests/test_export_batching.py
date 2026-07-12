import io
from uuid import uuid4

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest
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


async def test_export_streams_multiple_batches_not_one_shot_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = _FakeObjectStore()
    table = pa.table({"id": list(range(25)), "value": [float(i) for i in range(25)]})
    buf = io.BytesIO()
    pq.write_table(table, buf)
    key = "datasets/d/v/artifact.parquet"
    await store.put(key, buf.getvalue())
    version = DatasetVersion(
        id=uuid4(), tenant_id=uuid4(), dataset_id=uuid4(), artifact_uri=key, row_count=25
    )

    batch_sizes_seen: list[int] = []
    original_iter_batches = pq.ParquetFile.iter_batches

    def _spy_iter_batches(self, batch_size=65536, **kwargs):  # type: ignore[no-untyped-def]
        batches = list(original_iter_batches(self, batch_size=batch_size, **kwargs))
        batch_sizes_seen.append(len(batches))
        return iter(batches)

    monkeypatch.setattr(pq.ParquetFile, "iter_batches", _spy_iter_batches)

    artifact = await PyArrowExporter().export(version, store, format="csv", batch_size=10)

    assert artifact.row_count == 25
    # 25 rows at batch_size=10 -> 3 batches (10, 10, 5): proves chunked reading, not a
    # single pq.read_table() call.
    assert batch_sizes_seen and batch_sizes_seen[0] == 3
