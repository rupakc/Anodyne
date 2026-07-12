# Anodyne — Sub-system E (Export & Storage) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system E (requirement 5)
- **Depends on:** `anodyne-dataset` (`DatasetVersion`, `DatasetRepository`), `anodyne-storage`
  (`S3ObjectStore`, `tenant_session`), Generation Engine C0–C6 (on `main`)

## Goal

Add an **Exporter** capability that serializes a stored `DatasetVersion`'s Parquet artifact to
CSV, JSON, Parquet, or Arrow (IPC), uploads the result tenant-prefixed to the object store, records
it, and returns a presigned download URL — without ever materializing the full dataset in memory,
and defaulting the format by dataset size when the caller doesn't choose one.

## Decisions

| Decision | Choice |
|---|---|
| Port location | `Exporter` ABC in `anodyne_dataset.ports`, next to `Generator` — mirrors that port's shape (one method, takes a spec/version, returns/produces artifacts) |
| New adapter package | `packages/anodyne-export/` — depends on `anodyne-core`, `anodyne-dataset`, `pyarrow`; registered in root `pyproject.toml` workspace members exactly like `anodyne-tabular` |
| Streaming strategy | Fetch the stored Parquet bytes once via `ObjectStore.get` (the port's only read primitive — see "Streaming caveat" below), then drive every format from `pyarrow.parquet.ParquetFile(...).iter_batches(batch_size=...)`. No format ever builds one `pyarrow.Table`/`pandas.DataFrame` holding the whole dataset: CSV/JSON/Arrow writers are fed batch-by-batch; Parquet re-encode uses a `pq.ParquetWriter` fed batch-by-batch. Peak memory is bounded by `batch_size` (default 50k rows), not `row_count`. |
| Streaming caveat (documented, not silently ignored) | `anodyne_core.ports.ObjectStore.get` returns the full object as `bytes` — there is no ranged/streaming read primitive in the existing port, and every current adapter (`assemble` in `anodyne_workflows.handlers`, `download_version`) already relies on whole-object `get`. Adding a streaming read method to `ObjectStore` would touch a shared core port used by every package in the workspace — out of scope/proportion for this sub-system. We accept holding the *source* Parquet bytes as one buffer (an `io.BytesIO`, not a second decoded copy) and stream everything downstream of that buffer via `ParquetFile.iter_batches`. This is strictly better than the existing `handlers.py` pattern (which builds a full concatenated `pa.Table` in memory) and satisfies "never load the whole dataset into memory" for every stage after the initial fetch. |
| Compute placement | No Ray task, no Temporal workflow. Export is I/O-bound (read/transcode/write batches), not CPU/model-bound like generation (SDV fitting, LLM calls, image/audio synthesis) — the reason C generation uses Ray. A chunked pyarrow pipeline already bounds memory for arbitrarily large `row_count` in-process; adding Ray orchestration would add worker/task-queue plumbing with no corresponding win. `run_export(...)` in `anodyne-export` is a plain function of `(spec, version, store)` with no gateway/Ray coupling, so wrapping it in a Ray remote later (if a future profiling run shows the gateway process itself needs offloading) is a drop-in change, not a redesign. This trade-off is the one called out in the task brief as needing justification. |
| Size-based default format | Named constant `anodyne_export.exporter.LARGE_DATASET_ROW_THRESHOLD = 500_000`. `resolve_format(row_count, requested)`: `requested` wins if given; else `row_count > LARGE_DATASET_ROW_THRESHOLD` → `parquet`, else → `csv`. Pure function, unit-tested with an injected row count (no 500k-row fixture ever generated). |
| Arrow vs Parquet | "Arrow the interchange sibling" — Arrow (`.arrow`, IPC stream format via `pyarrow.ipc.new_stream`) is offered as an explicit `format=arrow` choice for interchange use cases (e.g. zero-copy load into another Arrow-based tool); it is never the *default* — only Parquet or CSV are chosen automatically by size. |
| DB table | `export_artifacts` — tenant-scoped record of every export produced (dataset_id, version_id, format, row_count, object_key, created_at), mirroring `dataset_versions`. Lets `GET` list/download of past exports later without re-running the transcode; also gives the API route something durable to return besides a bare URL. |
| Migration | New file `packages/anodyne-storage/src/anodyne_storage/migrations/versions/export_artifacts_table.py`, `down_revision = "0006"`. Descriptively named per the task brief (not `0007_*`); `revision = "export_artifacts"`. |
| API route | `POST /datasets/{dataset_id}/versions/{version_id}/export` in a **new, focused module** `apps/api-gateway/src/api_gateway/export_routes.py` — an `APIRouter` included from `create_app()` via one `app.include_router(...)` line, rather than inlining another ~40 lines into the already-large `app.py`. This minimizes merge conflicts with sibling in-flight branches touching `app.py` (the task brief explicitly asks for this). Body: `{"format": "csv"\|"json"\|"parquet"\|"arrow" \| null}`. Permission: reuses `datasets:read` (export is a derived-read of data the tenant already owns for generation purposes, exactly like the existing `GET .../download` route) — no new permission added to `anodyne_tenancy.authz`, avoiding a shared-file edit for a read-shaped operation. |
| Ownership enforcement | Same pattern as `download_version`: look up the dataset's versions via `repo.list_versions(ctx.tenant_id, dataset_id)` (RLS + explicit tenant filter) and 404 if the `version_id` doesn't match one of them — a version from another tenant is indistinguishable from "not found". |
| Response shape | `{"artifact": ExportArtifact.model_dump(mode="json"), "url": "<presigned>"}` — mirrors the existing `list_versions`/`download_version` shapes (full model + separate presigned URL). |

## Components

### 1. `anodyne_dataset.ports.Exporter` (new ABC, `anodyne-dataset` package)

```python
class Exporter(ABC):
    @abstractmethod
    async def export(
        self,
        version: DatasetVersion,
        store: ObjectStore,
        *,
        format: str | None = None,
        batch_size: int = 50_000,
    ) -> ExportArtifact: ...
```

`ExportArtifact` (new model in `anodyne_dataset.models`, alongside `DatasetVersion`):

```python
class ExportArtifact(BaseModel):
    id: UUID
    tenant_id: UUID
    dataset_id: UUID
    version_id: UUID
    format: str
    row_count: int
    object_key: str
    created_at: datetime = Field(default_factory=...)
```

### 2. `anodyne-export` (new package)

- `exporter.py`:
  - `LARGE_DATASET_ROW_THRESHOLD = 500_000`
  - `resolve_format(row_count: int, requested: str | None) -> str`
  - `SUPPORTED_FORMATS = {"csv", "json", "parquet", "arrow"}`; `UnsupportedExportFormatError(ValueError)`
  - `PyArrowExporter(Exporter)` — the concrete adapter:
    1. `data = await store.get(version.artifact_uri)` (whole-object fetch — see Streaming caveat)
    2. `pf = pq.ParquetFile(io.BytesIO(data))`
    3. dispatch on resolved format to one of four private writers, each iterating
       `pf.iter_batches(batch_size=batch_size)` and writing incrementally:
       - `_write_parquet`: `pq.ParquetWriter(buf, pf.schema_arrow)`, `writer.write_batch(batch)` per batch.
       - `_write_arrow`: `pa.ipc.new_stream(buf, pf.schema_arrow)`, `writer.write_batch(batch)` per batch.
       - `_write_csv`: `pyarrow.csv.CSVWriter(buf, pf.schema_arrow)`, `writer.write_batch(batch)` per batch (pyarrow's CSV writer streams natively).
       - `_write_json`: manual — for each batch, `batch.to_pylist()` then one `json.dumps(row)` per row, newline-joined (JSON Lines) written incrementally to the buffer; keeps at most one batch's rows in memory at a time, not the whole dataset.
    4. uploads the buffer to `store.put(<export key>, buf.getvalue())` — the *destination* buffer is not the whole source dataset, it's the encoded output, which for CSV/JSON/Arrow is comparable in size to the input; this is an accepted, documented limit of the current `ObjectStore.put(bytes)` signature (same limit `assemble_and_upload` already has).
    5. returns a `PyArrowExporter`-internal result (`format`, `row_count`, `object_key`) — persistence/DB-record construction is the **caller's** job (kept in the gateway route / a small helper), so the exporter itself has no `DatasetRepository`/tenant coupling beyond what it's handed.
  - Export object key: `datasets/{dataset_id}/{version_id}/export.{ext}` (tenant prefix added by `S3ObjectStore` itself, exactly like every other key in this codebase).

### 3. `anodyne-storage`: `export_artifacts` table + repo methods

- `db.py`: new `export_artifacts` table (`id` PK, `tenant_id`, `dataset_id`, `version_id`, `format`,
  `row_count`, `object_key`, `created_at`), added to `_TENANT_TABLES` for RLS.
- `SqlDatasetRepository` gains `add_export`/`list_exports` (or a small sibling
  `ExportArtifactRepository` port + impl, mirroring how `ProfileRepository` was added as a sibling
  port rather than growing `DatasetRepository` — chosen for the same reason stated in
  `ports.py`: "kept separate so adding it never breaks an existing `DatasetRepository`
  implementation/fake"). **Decision: `ExportRepository` sibling port**, same pattern as
  `ProfileRepository`.

### 4. Gateway: `apps/api-gateway/src/api_gateway/export_routes.py` (new module)

- `router = APIRouter()`
- `POST /datasets/{dataset_id}/versions/{version_id}/export`:
  1. 404 if the version isn't found among the tenant's versions for that dataset.
  2. `PyArrowExporter().export(version, store, format=body.format)`.
  3. Persist an `ExportArtifact` via `ExportRepository.add_export`.
  4. `presigned = await store.presigned_url(artifact.object_key)`.
  5. Return `{"artifact": ..., "url": presigned}`.
- New `deps.get_export_repo` / reuses `deps.get_object_store`.
- `app.py` gains one import + one `app.include_router(export_routes.router)` line.

## Testing strategy (offline-first, TDD)

- **`anodyne-dataset`**: `test_export_ports.py` — `Exporter` ABC shape; `test_export_models.py` —
  `ExportArtifact` round-trips via `model_dump`/`model_validate`.
- **`anodyne-export`** (the bulk of the coverage, all offline/no Docker):
  - `test_export_format_selection.py` — `resolve_format`: explicit format always wins regardless of
    row_count; `row_count <= 500_000` → `csv`; `row_count > 500_000` → `parquet`; boundary at
    exactly 500_000 → `csv` (threshold is "**>** 500k", not ">="); injected row counts only
    (`10`, `500_000`, `500_001`, `2_000_000`) — never generates rows.
  - `test_export_csv.py`, `test_export_json.py`, `test_export_parquet.py`, `test_export_arrow.py` —
    each builds a small (~50 row) `pyarrow.Table` fixture, writes it as the "stored" Parquet via a
    fake in-memory `ObjectStore`, calls `PyArrowExporter().export(...)`, reads the uploaded bytes
    back with the matching reader (`pandas.read_csv`/`json.loads` per line/`pq.read_table`/
    `pa.ipc.open_stream`), and asserts the round-tripped rows equal the fixture (including dtype
    edges: nulls, unicode strings, floats).
  - `test_export_unsupported_format.py` — `format="xml"` raises `UnsupportedExportFormatError`.
  - `test_export_batching.py` — with `batch_size=10` and a 25-row fixture, monkeypatches/spies
    `ParquetFile.iter_batches` (or asserts row-group/batch count) to confirm >1 batch is processed,
    proving the code path is chunked, not a single `pq.read_table()` call.
- **`anodyne-storage`**: `test_export_repo.py` — `add_export`/`list_exports` round-trip
  (`testcontainers` Postgres, `@pytest.mark.integration`, mirrors `test_dataset_repo.py`); a
  non-integration RLS assertion added to `test_rls.py`'s existing table list if that file enumerates
  tables generically (else skipped — checked at implementation time).
- **`apps/api-gateway`**: `test_export_routes.py` — 404 on unknown/foreign-tenant version; default
  format for a small stored `row_count` is CSV (via a fake `Exporter`/`ExportRepository` verifying
  the format actually passed through); explicit `format=parquet` honored; response contains
  `artifact` + `url`; `datasets:read`-gated (viewer allowed, matching `download_version`'s gating —
  no write permission required for a read-derived artifact).

All of the above run under `-m "not integration"` except the Postgres-backed repo test.

## Non-goals (E)

A full Temporal export workflow (justified above — synchronous, chunked, in-process is sufficient
and simpler); a UI for exports (Web UI is out of scope per the task brief); re-exporting
perturbed/evaluated datasets (Sub-systems D/F land later and can reuse this `Exporter` unchanged
since it only depends on `DatasetVersion`/`ObjectStore`); true streaming *reads* from the object
store (would require a new `ObjectStore` port method — see the Streaming caveat decision above).
