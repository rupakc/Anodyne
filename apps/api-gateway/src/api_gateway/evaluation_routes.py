"""Evaluation Engine (sub-system F) API surface.

A focused `APIRouter` mounted by `create_app`: launch a 360-degree MoE
LLM-as-a-Judge evaluation of a dataset version, poll its status, and fetch/
download the resulting report artifact. Tenant ownership is enforced by
resolving every version/run through the caller's own tenant-scoped repositories
(RLS + explicit `tenant_id` filter), so one tenant can never evaluate or read
another tenant's data by guessing ids.
"""

from __future__ import annotations

import json
from uuid import UUID, uuid4

# Import for its side effect: populates the `task_metrics` provider registry
# (catalog_for/provider_for) with every task-type provider, so the
# task-metrics catalog route and the evaluation workflow both see the full
# set regardless of which module happened to import them first.
import anodyne_evaluation.judges.task_metrics  # noqa: F401
from anodyne_core.models import TenantContext
from anodyne_core.ports import ObjectStore
from anodyne_dataset.ports import DatasetRepository
from anodyne_evaluation.models import EvaluationRun
from anodyne_evaluation.ports import EvaluationRepository
from anodyne_evaluation.task import TaskType, detect_task
from anodyne_evaluation.task_metrics import catalog_for
from anodyne_workflows.evaluation_workflow import EvaluationInput, EvaluationWorkflow
from fastapi import APIRouter, Depends, HTTPException, Query, Response
from pydantic import BaseModel, field_validator
from temporalio.client import Client

from api_gateway import deps
from api_gateway.config import get_settings
from api_gateway.downloads import content_disposition


class EvaluateRequest(BaseModel):
    reference_version_id: UUID | None = None
    seed: int = 0
    sensitive_field: str | None = None
    target_field: str | None = None
    text_column: str | None = None
    model_config_id: UUID | None = None
    sample_rows: int = 20
    weights: dict[str, float] = {}
    selected_metrics: list[str] | None = None
    task_type: str | None = None

    @field_validator("task_type")
    @classmethod
    def _validate_task_type(cls, v: str | None) -> str | None:
        if v is not None and v not in set(TaskType):
            valid = [t.value for t in TaskType]
            raise ValueError(f"invalid task_type {v!r}; expected one of {valid}")
        return v


def _build_config(body: EvaluateRequest) -> dict[str, object]:
    cfg: dict[str, object] = {
        "seed": body.seed,
        "sample_rows": body.sample_rows,
        "weights": body.weights,
    }
    if body.sensitive_field is not None:
        cfg["sensitive_field"] = body.sensitive_field
    if body.target_field is not None:
        cfg["target_field"] = body.target_field
    if body.text_column is not None:
        cfg["text_column"] = body.text_column
    if body.model_config_id is not None:
        cfg["model_config_id"] = str(body.model_config_id)
    if body.selected_metrics is not None:
        cfg["selected_metrics"] = body.selected_metrics
    if body.task_type is not None:
        cfg["task_type"] = body.task_type
    return cfg


def build_router() -> APIRouter:
    router = APIRouter()

    @router.post("/datasets/{dataset_id}/versions/{version_id}/evaluate", status_code=202)
    async def start_evaluation(
        dataset_id: UUID,
        version_id: UUID,
        body: EvaluateRequest,
        ctx: TenantContext = Depends(deps.require("evaluations:write")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
        eval_repo: EvaluationRepository = Depends(deps.get_evaluation_repo),
        client: Client = Depends(deps.get_temporal_client),
        registry: deps.ModelRegistry = Depends(deps.get_model_registry),
    ) -> dict[str, object]:
        versions = await repo.list_versions(ctx.tenant_id, dataset_id)
        ids = {v.id for v in versions}
        if version_id not in ids:
            raise HTTPException(404, "dataset version not found")
        if body.reference_version_id is not None and body.reference_version_id not in ids:
            raise HTTPException(404, "reference version not found")

        # An explicit `model_config_id` always wins; otherwise resolve the
        # tenant's default (prefers the configured `default_llm_provider`,
        # e.g. Gemini, over whatever config happens to be registered first --
        # see `deps.pick_default_model`). No crash if the tenant has no
        # models registered at all: the qualitative judge just runs without
        # one (mirrors the pre-existing behavior for that case).
        if body.model_config_id is None:
            configs = await registry.list(ctx.tenant_id)
            if configs:
                body.model_config_id = deps.pick_default_model(
                    configs, get_settings().default_llm_provider
                ).id

        run = EvaluationRun(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            dataset_id=dataset_id,
            dataset_version_id=version_id,
            reference_version_id=body.reference_version_id,
            config=_build_config(body),
        )
        handle = await client.start_workflow(
            EvaluationWorkflow.run,
            EvaluationInput(
                run_id=str(run.id),
                dataset_id=str(dataset_id),
                tenant_id=str(ctx.tenant_id),
                dataset_version_id=str(version_id),
                reference_version_id=(
                    str(body.reference_version_id) if body.reference_version_id else None
                ),
                seed=body.seed,
                config=run.config,
            ),
            id=f"eval-{run.id}",
            task_queue="evaluation",
        )
        run.workflow_id = handle.id
        await eval_repo.create_run(run)
        return run.model_dump(mode="json")

    @router.get("/datasets/{dataset_id}/versions/{version_id}/task-metrics")
    async def get_task_metrics(
        dataset_id: UUID,
        version_id: UUID,
        ctx: TenantContext = Depends(deps.require("evaluations:read")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
    ) -> dict[str, object]:
        version = await repo.get_version(ctx.tenant_id, version_id)
        if version is None or version.dataset_id != dataset_id:
            raise HTTPException(404, "dataset version not found")
        spec = await repo.get_spec(ctx.tenant_id, dataset_id)
        if spec is None:
            raise HTTPException(404, "dataset spec not found")
        columns = [f.name for f in spec.fields]
        task_type = detect_task(spec.modality, columns)
        return {
            "task_type": task_type.value,
            "available_metrics": [m.model_dump(mode="json") for m in catalog_for(task_type)],
        }

    @router.get("/evaluations/{run_id}")
    async def get_evaluation(
        run_id: UUID,
        ctx: TenantContext = Depends(deps.require("evaluations:read")),
        eval_repo: EvaluationRepository = Depends(deps.get_evaluation_repo),
    ) -> dict[str, object]:
        run = await eval_repo.get_run(ctx.tenant_id, run_id)
        if run is None:
            raise HTTPException(404, "evaluation not found")
        return run.model_dump(mode="json")

    @router.get("/evaluations/{run_id}/report")
    async def get_evaluation_report(
        run_id: UUID,
        ctx: TenantContext = Depends(deps.require("evaluations:read")),
        eval_repo: EvaluationRepository = Depends(deps.get_evaluation_repo),
        object_store: ObjectStore = Depends(deps.get_object_store),
    ) -> dict[str, object]:
        run = await eval_repo.get_run(ctx.tenant_id, run_id)
        if run is None:
            raise HTTPException(404, "evaluation not found")
        if run.report_uri is None:
            raise HTTPException(409, "evaluation report not ready")
        data = await object_store.get(run.report_uri)
        return json.loads(data)  # type: ignore[no-any-return]

    @router.get("/evaluations/{run_id}/report/download")
    async def download_evaluation_report(
        run_id: UUID,
        format: str = Query("html"),
        ctx: TenantContext = Depends(deps.require("evaluations:read")),
        eval_repo: EvaluationRepository = Depends(deps.get_evaluation_repo),
        object_store: ObjectStore = Depends(deps.get_object_store),
    ) -> Response:
        if format not in ("html", "json"):
            raise HTTPException(400, f"unsupported format {format!r}; expected 'html' or 'json'")
        run = await eval_repo.get_run(ctx.tenant_id, run_id)
        if run is None:
            raise HTTPException(404, "evaluation not found")
        uri = run.report_html_uri if format == "html" else run.report_uri
        if uri is None:
            raise HTTPException(409, "evaluation report not ready")
        # Stream the report bytes through the gateway rather than a presigned
        # URL -- see `download_version` in `app.py` for why.
        data = await object_store.get(uri)
        media_type = "text/html" if format == "html" else "application/json"
        return Response(
            content=data,
            media_type=media_type,
            headers=content_disposition(f"evaluation-{run_id}.{format}"),
        )

    return router
