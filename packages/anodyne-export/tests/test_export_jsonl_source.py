"""Export a text dataset whose stored artifact is JSONL (not Parquet).

Regression: the exporter read every source artifact as Parquet, so exporting a
text version (`format="jsonl"`) to CSV/JSON/Parquet/Arrow raised
`pyarrow.lib.ArrowInvalid: Parquet magic bytes not found` -> HTTP 500 ->
"Failed to Fetch" in the UI.
"""

import io
import json
from uuid import uuid4

import pandas as pd  # type: ignore[import-untyped]
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


_ROWS = [
    {"text": "Just finished my marathon! #Achievement", "label": "positive"},
    {"text": "Ugh, my internet has been down all morning.", "label": "negative"},
    {"text": "The library announced new extended hours.", "label": "neutral"},
]


async def _stored_jsonl_version(store: _FakeObjectStore) -> DatasetVersion:
    payload = "\n".join(json.dumps(r) for r in _ROWS).encode("utf-8")
    key = "datasets/d/v/artifact.jsonl"
    await store.put(key, payload)
    return DatasetVersion(
        id=uuid4(),
        tenant_id=uuid4(),
        dataset_id=uuid4(),
        artifact_uri=key,
        format="jsonl",
        row_count=len(_ROWS),
    )


async def test_csv_export_from_jsonl_source() -> None:
    store = _FakeObjectStore()
    version = await _stored_jsonl_version(store)

    artifact = await PyArrowExporter().export(version, store, format="csv")

    assert artifact.format == "csv"
    assert artifact.row_count == len(_ROWS)
    restored = pd.read_csv(io.BytesIO(await store.get(artifact.object_key)))
    assert restored["label"].tolist() == [r["label"] for r in _ROWS]
    assert restored["text"].tolist() == [r["text"] for r in _ROWS]


async def test_json_export_from_jsonl_source() -> None:
    store = _FakeObjectStore()
    version = await _stored_jsonl_version(store)

    artifact = await PyArrowExporter().export(version, store, format="json")

    assert artifact.row_count == len(_ROWS)
    lines = (await store.get(artifact.object_key)).decode().splitlines()
    assert [json.loads(x) for x in lines] == _ROWS
