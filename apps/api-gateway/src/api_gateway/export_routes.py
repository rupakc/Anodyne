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
from anodyne_export.exporter import SUPPORTED_FORMATS
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from api_gateway import deps
from api_gateway.downloads import content_disposition, media_type_and_ext, safe_filename

router = APIRouter()


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

    artifact = await exporter.export(version, object_store, format=body.format)
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
