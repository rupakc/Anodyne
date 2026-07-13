"""Sub-system E: `POST /datasets/{dataset_id}/versions/{version_id}/export`.

Kept in its own `APIRouter` (rather than inlined into the already-large `app.py`) so this
sub-system's routes are additive to `app.py` with a single `include_router` line -- minimizing
merge conflicts with sibling in-flight branches also touching `app.py`.

Permission: reuses `datasets:read` (no new permission added to `anodyne_tenancy.authz`). Export is
a derived-read of data the tenant already owns for generation purposes, exactly like the existing
`GET /datasets/{id}/versions/{id}/download` route -- not a mutation of the dataset itself.
"""

from __future__ import annotations

from uuid import UUID

from anodyne_core.models import TenantContext
from anodyne_core.ports import ObjectStore
from anodyne_dataset.ports import DatasetRepository, Exporter, ExportRepository
from anodyne_export.exporter import SUPPORTED_FORMATS as TABULAR_SUPPORTED_FORMATS
from anodyne_export.exporter import UnsupportedExportFormatError
from anodyne_graph.errors import UnsupportedGraphExportFormatError
from anodyne_graph.export import GRAPH_SUPPORTED_FORMATS
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from api_gateway import deps
from api_gateway.downloads import content_disposition, media_type_and_ext, safe_filename

router = APIRouter()

# GC: the union of tabular (pyarrow) and graph formats -- request-level validation doesn't yet
# know the version's modality (that requires a repo lookup), so any format legal for *either*
# exporter passes this gate; `export_version` then dispatches to the exporter that actually
# understands the version's artifact and re-validates against its own narrower format set.
SUPPORTED_FORMATS = TABULAR_SUPPORTED_FORMATS | GRAPH_SUPPORTED_FORMATS

# `DatasetVersion.format`/`GraphHandler.artifact_format` value that marks a graph-modality
# artifact (see `anodyne_graph.serialization` -- the node-link JSON contract). Any other
# version format is routed to the tabular `PyArrowExporter`.
_GRAPH_ARTIFACT_FORMAT = "graph_json"


class ExportRequest(BaseModel):
    format: str | None = None


@router.post("/datasets/{dataset_id}/versions/{version_id}/export")
async def export_version(
    dataset_id: UUID,
    version_id: UUID,
    body: ExportRequest,
    ctx: TenantContext = Depends(deps.require("datasets:read")),
    repo: DatasetRepository = Depends(deps.get_dataset_repo),
    export_repo: ExportRepository = Depends(deps.get_export_repo),
    exporter: Exporter = Depends(deps.get_exporter),
    graph_exporter: Exporter = Depends(deps.get_graph_exporter),
    object_store: ObjectStore = Depends(deps.get_object_store),
) -> Response:
    if body.format is not None and body.format not in SUPPORTED_FORMATS:
        raise HTTPException(
            400, f"unsupported format {body.format!r}; expected one of {sorted(SUPPORTED_FORMATS)}"
        )

    versions = await repo.list_versions(ctx.tenant_id, dataset_id)
    version = next((v for v in versions if v.id == version_id), None)
    if version is None:
        raise HTTPException(404, "version not found")

    # GC dispatch: a graph-modality version's stored artifact is always `graph_json` (see
    # `GraphHandler.artifact_format`) -- route it to `GraphExporter`, never the pyarrow one.
    active_exporter = graph_exporter if version.format == _GRAPH_ARTIFACT_FORMAT else exporter
    try:
        artifact = await active_exporter.export(version, object_store, format=body.format)
    except (UnsupportedExportFormatError, UnsupportedGraphExportFormatError) as exc:
        # `SUPPORTED_FORMATS` above is the *union* of tabular + graph formats (checked before the
        # version's modality is known); a format valid for one exporter but not the other's own
        # narrower set surfaces here as a 400, not an unhandled 500.
        raise HTTPException(400, str(exc)) from exc
    await export_repo.add_export(artifact)

    # Stream the just-created artifact through the gateway rather than
    # handing back a presigned URL -- see `download_version` in `app.py` for
    # why (presigned URLs go stale across an open page/sleep/clock jump).
    data = await object_store.get(artifact.object_key)
    media_type, ext = media_type_and_ext(artifact.format)
    spec = await repo.get_spec(ctx.tenant_id, dataset_id)
    base_name = safe_filename(spec.name if spec is not None else str(dataset_id))
    return Response(
        content=data,
        media_type=media_type,
        headers=content_disposition(f"{base_name}.{ext}"),
    )
