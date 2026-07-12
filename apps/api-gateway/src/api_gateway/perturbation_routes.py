"""Perturbation API (sub-system D): launch a perturbation on a stored
`DatasetVersion` and inspect the resulting jobs.

A focused `APIRouter` included by `create_app`, mirroring the generation routes'
shape (tenant ownership checks, `deps.require(...)` permissions, Temporal
`start_workflow`). Every read/write is tenant-scoped: routes resolve the
dataset/version/job through the tenant-filtered repositories, so one tenant can
never perturb or read another tenant's data.
"""

from __future__ import annotations

from uuid import UUID, uuid4

from anodyne_core.models import TenantContext
from anodyne_dataset.models import (
    PerturbationFamily,
    PerturbationJob,
    PerturbationSpec,
)
from anodyne_dataset.ports import DatasetRepository, PerturbationRepository
from anodyne_workflows.perturbation_workflow import PerturbationInput, PerturbationWorkflow
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from temporalio.client import Client

from api_gateway import deps

router = APIRouter()


class PerturbRequest(BaseModel):
    family: PerturbationFamily
    intensity: float = Field(default=0.1, ge=0.0, le=1.0)
    target_fields: list[str] = Field(default_factory=list)
    params: dict[str, object] = Field(default_factory=dict)
    seed: int = 0


@router.post("/datasets/{dataset_id}/versions/{version_id}/perturb", status_code=202)
async def launch_perturbation(
    dataset_id: UUID,
    version_id: UUID,
    body: PerturbRequest,
    ctx: TenantContext = Depends(deps.require("perturbations:write")),
    repo: DatasetRepository = Depends(deps.get_dataset_repo),
    pert_repo: PerturbationRepository = Depends(deps.get_perturbation_repo),
    client: Client = Depends(deps.get_temporal_client),
) -> dict[str, object]:
    spec = await repo.get_spec(ctx.tenant_id, dataset_id)
    if spec is None:
        raise HTTPException(404, "dataset not found")
    versions = await repo.list_versions(ctx.tenant_id, dataset_id)
    if not any(v.id == version_id for v in versions):
        raise HTTPException(404, "version not found")

    pert_spec = PerturbationSpec(
        family=body.family,
        intensity=body.intensity,
        target_fields=body.target_fields,
        params=body.params,
        seed=body.seed,
    )
    job = PerturbationJob(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        dataset_id=dataset_id,
        parent_version_id=version_id,
        spec=pert_spec,
    )
    handle = await client.start_workflow(
        PerturbationWorkflow.run,
        PerturbationInput(
            job_id=str(job.id),
            dataset_id=str(dataset_id),
            tenant_id=str(ctx.tenant_id),
            parent_version_id=str(version_id),
            family=body.family.value,
            intensity=body.intensity,
            seed=body.seed,
            params=body.params,
            target_fields=body.target_fields,
            modality=spec.modality.value,
        ),
        id=f"pert-{job.id}",
        task_queue="generation",
    )
    job.workflow_id = handle.id
    await pert_repo.save_perturbation_job(job)
    return job.model_dump(mode="json")


@router.get("/perturbation-jobs/{job_id}")
async def get_perturbation_job(
    job_id: UUID,
    ctx: TenantContext = Depends(deps.require("perturbations:read")),
    pert_repo: PerturbationRepository = Depends(deps.get_perturbation_repo),
) -> dict[str, object]:
    job = await pert_repo.get_perturbation_job(ctx.tenant_id, job_id)
    if job is None:
        raise HTTPException(404, "perturbation job not found")
    return job.model_dump(mode="json")


@router.get("/datasets/{dataset_id}/perturbation-jobs")
async def list_perturbation_jobs(
    dataset_id: UUID,
    ctx: TenantContext = Depends(deps.require("perturbations:read")),
    pert_repo: PerturbationRepository = Depends(deps.get_perturbation_repo),
) -> list[dict[str, object]]:
    jobs = await pert_repo.list_perturbation_jobs(ctx.tenant_id, dataset_id)
    return [j.model_dump(mode="json") for j in jobs]
