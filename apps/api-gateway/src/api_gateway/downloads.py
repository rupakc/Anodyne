"""Shared helpers for streaming artifact bytes through the gateway as a download.

Centralized here (rather than duplicated across `app.py`, `export_routes.py`, and
`evaluation_routes.py`) so every download route derives filenames/content-types the
same way. These routes stream bytes fetched via `ObjectStore.get` directly in the
response body with a `Content-Disposition: attachment` header -- deliberately NOT a
presigned MinIO/S3 URL, which can go stale ("Signature has expired") if the browser
holds onto it across a page staying open, a laptop sleep, or a clock jump.
"""

from __future__ import annotations

import re

# Dataset/export artifact format -> (media type, file extension). Keys match
# `DatasetVersion.format`/`ExportArtifact.format` (see anodyne_dataset.models) and
# `anodyne_export.exporter.SUPPORTED_FORMATS`.
_FORMAT_MEDIA_TYPES: dict[str, tuple[str, str]] = {
    "parquet": ("application/octet-stream", "parquet"),
    "arrow": ("application/vnd.apache.arrow.file", "arrow"),
    "csv": ("text/csv", "csv"),
    "json": ("application/json", "jsonl"),
    "jsonl": ("application/x-ndjson", "jsonl"),
    # Graph modality: node-link JSON (see anodyne_graph.serialization).
    "graph_json": ("application/json", "json"),
}

_DEFAULT_MEDIA_TYPE = "application/octet-stream"

# Anything other than word chars/dash/underscore/dot collapses to "_" so the
# resulting name is always a safe, single-segment filename (no path
# separators, no header-splitting characters).
_UNSAFE_FILENAME_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def media_type_and_ext(format: str) -> tuple[str, str]:
    """Map a dataset/export artifact `format` to `(media_type, file_extension)`.

    Falls back to `application/octet-stream` + the format string itself as the
    extension for any format not in the known table, so an unexpected/future
    format never raises -- it just downloads generically.
    """
    return _FORMAT_MEDIA_TYPES.get(format, (_DEFAULT_MEDIA_TYPE, format))


def safe_filename(name: str) -> str:
    """Sanitize a user-provided name (e.g. a dataset name) for use in a
    `Content-Disposition` filename: collapses anything not alnum/dot/dash/underscore
    to "_" and falls back to "download" if that leaves nothing usable.
    """
    cleaned = _UNSAFE_FILENAME_CHARS.sub("_", name).strip("._") or "download"
    return cleaned


def content_disposition(filename: str) -> dict[str, str]:
    """Build the single `Content-Disposition` response header for an attachment
    download with the given (already-sanitized) filename."""
    return {"Content-Disposition": f'attachment; filename="{filename}"'}
