"""`PyArrowExporter`: transcodes a stored `DatasetVersion` artifact to another format.

The stored artifact is read according to the version's own `format` -- Parquet for tabular
datasets, JSON Lines for text datasets -- via `_read_source`, which yields a uniform
`_BatchSource`. Everything downstream (schema, batching, transcode) is source-format-agnostic.

Formats supported: CSV, JSON (lines), Parquet, Arrow (IPC stream).

Memory profile: every writer is fed `pyarrow.parquet.ParquetFile.iter_batches(batch_size=...)`
batches rather than a single materialized `pyarrow.Table`/`pandas.DataFrame`, and the *encoded
output* is streamed into a `tempfile.SpooledTemporaryFile` that spills to disk past
`_SPILL_THRESHOLD` -- it is never buffered whole in memory. The output is then uploaded straight
from that file handle via `ObjectStore.put_fileobj`. So the transcode's peak memory is bounded by
`batch_size` (plus the small in-RAM head of the spool), not by the dataset's total row count.

The one unavoidable in-memory copy is the initial fetch of the *source* Parquet bytes via
`ObjectStore.get` (the port's only read primitive returns a whole `bytes` object) -- see the
"Streaming caveat" in `docs/superpowers/specs/2026-07-12-export-e-design.md` for why that isn't
solved here.
"""

from __future__ import annotations

import io
import json
import tempfile
import uuid
from collections.abc import Iterator
from typing import IO, Any, Protocol

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.csv as pcsv  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from anodyne_core.ports import ObjectStore
from anodyne_dataset.models import DatasetVersion, ExportArtifact
from anodyne_dataset.ports import Exporter

# Requirement 5's size rule: datasets over this many rows default to Parquet when the caller
# doesn't request a format; at or under it, default to CSV. Named constant per the spec.
LARGE_DATASET_ROW_THRESHOLD = 500_000

SUPPORTED_FORMATS = frozenset({"csv", "json", "parquet", "arrow"})

_DEFAULT_BATCH_SIZE = 50_000

# Encoded output stays in memory up to this size, then the spool rolls over to a real temp file on
# disk -- keeping small exports fast while bounding peak memory for large ones.
_SPILL_THRESHOLD = 32 * 1024 * 1024  # 32 MiB

_EXTENSIONS = {"csv": "csv", "json": "json", "parquet": "parquet", "arrow": "arrow"}


def _spool() -> IO[bytes]:
    return tempfile.SpooledTemporaryFile(max_size=_SPILL_THRESHOLD, mode="w+b")


class UnsupportedExportFormatError(ValueError):
    """Raised when `format` isn't one of `SUPPORTED_FORMATS`."""


class UnsupportedSourceFormatError(ValueError):
    """Raised when the stored artifact's own format can't be read for export."""


class _BatchSource(Protocol):
    """The uniform view the writers consume, regardless of the stored artifact's
    on-disk format: a schema, a row count, and a batch iterator."""

    @property
    def schema_arrow(self) -> Any: ...

    @property
    def num_rows(self) -> int: ...

    def iter_batches(self, batch_size: int) -> Iterator[Any]: ...


class _ParquetSource:
    """Tabular artifacts: stream batches straight off the Parquet file."""

    def __init__(self, data: bytes) -> None:
        self._pf = pq.ParquetFile(io.BytesIO(data))

    @property
    def schema_arrow(self) -> Any:
        return self._pf.schema_arrow

    @property
    def num_rows(self) -> int:
        return int(self._pf.metadata.num_rows)

    def iter_batches(self, batch_size: int) -> Iterator[Any]:
        return iter(self._pf.iter_batches(batch_size=batch_size))


class _TableSource:
    """Row-oriented artifacts (e.g. text `jsonl`): parsed into an in-memory table
    and re-chunked. The one materialization is bounded by the artifact `get`
    already fetching the whole `bytes` (see the module docstring's caveat)."""

    def __init__(self, table: Any) -> None:
        self._table = table

    @property
    def schema_arrow(self) -> Any:
        return self._table.schema

    @property
    def num_rows(self) -> int:
        return int(self._table.num_rows)

    def iter_batches(self, batch_size: int) -> Iterator[Any]:
        return iter(self._table.to_batches(max_chunksize=batch_size))


def _read_source(data: bytes, fmt: str) -> _BatchSource:
    """Build a `_BatchSource` from stored artifact bytes according to the
    version's own `format` -- tabular is Parquet, text is JSON Lines. Everything
    downstream (schema, batching, transcode) is format-agnostic from here."""
    if fmt == "parquet":
        return _ParquetSource(data)
    if fmt == "jsonl":
        rows = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        return _TableSource(pa.Table.from_pylist(rows))
    raise UnsupportedSourceFormatError(
        f"cannot export a source artifact of format {fmt!r}; expected 'parquet' or 'jsonl'"
    )


def resolve_format(row_count: int, requested: str | None) -> str:
    """Pick the export format: `requested` always wins; otherwise size-based default.

    `row_count > LARGE_DATASET_ROW_THRESHOLD` -> `"parquet"`; else `"csv"`. Pure function so the
    >500k-row branch is unit-testable with an injected row count, never a generated fixture.
    """
    if requested is not None:
        return requested
    return "parquet" if row_count > LARGE_DATASET_ROW_THRESHOLD else "csv"


def _artifact_key(dataset_id: uuid.UUID, version_id: uuid.UUID, ext: str) -> str:
    # Tenant-relative, exactly like generation's `_artifact_key`/`_shard_key`: the per-tenant
    # `S3ObjectStore` prepends `{tenant_id}/` itself, so this key must NOT repeat it. Handing the
    # exporter a tenant-scoped store is what lands the export under the tenant's prefix and makes
    # the presigned URL tenant-scoped.
    return f"datasets/{dataset_id}/{version_id}/export.{ext}"


def _write_parquet(source: _BatchSource, batch_size: int) -> IO[bytes]:
    out = _spool()
    writer = pq.ParquetWriter(out, source.schema_arrow)
    try:
        for batch in source.iter_batches(batch_size):
            writer.write_batch(batch)
    finally:
        writer.close()
    out.seek(0)
    return out


def _write_arrow(source: _BatchSource, batch_size: int) -> IO[bytes]:
    out = _spool()
    writer = pa.ipc.new_stream(out, source.schema_arrow)
    try:
        for batch in source.iter_batches(batch_size):
            writer.write_batch(batch)
    finally:
        writer.close()
    out.seek(0)
    return out


def _write_csv(source: _BatchSource, batch_size: int) -> IO[bytes]:
    out = _spool()
    writer = pcsv.CSVWriter(out, source.schema_arrow)
    try:
        for batch in source.iter_batches(batch_size):
            writer.write_batch(batch)
    finally:
        writer.close()
    out.seek(0)
    return out


def _write_json(source: _BatchSource, batch_size: int) -> IO[bytes]:
    # JSON Lines: one object per row, written batch-by-batch so at most one batch's rows are
    # ever materialized as Python objects at a time (not the whole dataset).
    out = _spool()
    for batch in source.iter_batches(batch_size):
        for row in batch.to_pylist():
            out.write(json.dumps(row).encode())
            out.write(b"\n")
    out.seek(0)
    return out


_WRITERS = {
    "parquet": _write_parquet,
    "arrow": _write_arrow,
    "csv": _write_csv,
    "json": _write_json,
}


class PyArrowExporter(Exporter):
    """The concrete, pyarrow-backed `Exporter` adapter."""

    async def export(
        self,
        version: DatasetVersion,
        store: ObjectStore,
        *,
        format: str | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
    ) -> ExportArtifact:
        resolved = resolve_format(version.row_count, format)
        if resolved not in SUPPORTED_FORMATS:
            raise UnsupportedExportFormatError(
                f"unsupported export format {resolved!r}; expected one of "
                f"{sorted(SUPPORTED_FORMATS)}"
            )

        data = await store.get(version.artifact_uri)
        source = _read_source(data, version.format)
        row_count = source.num_rows

        key = _artifact_key(version.dataset_id, version.id, _EXTENSIONS[resolved])
        encoded = _WRITERS[resolved](source, batch_size)
        try:
            await store.put_fileobj(key, encoded)
        finally:
            encoded.close()

        return ExportArtifact(
            id=uuid.uuid4(),
            tenant_id=version.tenant_id,
            dataset_id=version.dataset_id,
            version_id=version.id,
            format=resolved,
            row_count=row_count,
            object_key=key,
        )
