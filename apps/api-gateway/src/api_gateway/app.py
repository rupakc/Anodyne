from __future__ import annotations

import asyncio
import contextlib
from uuid import UUID, uuid4

from anodyne_core.models import LLMRequest, TenantContext
from anodyne_core.ports import LLMProvider, ObjectStore
from anodyne_dataset.models import DatasetSpec, FieldSpec, GenerationJob, Modality
from anodyne_dataset.ports import (
    DatasetRepository,
    ProfileRepository,
    SampleProfiler,
    SchemaProposer,
)
from anodyne_generation.proposer import SchemaProposalError
from anodyne_observability.logging import bind_request_context, configure_logging
from anodyne_tabular.io import UnsupportedSampleFormatError
from anodyne_tabular.schema import fields_from_profile
from anodyne_templates.catalog import build_dataset_spec, get_template, list_templates
from anodyne_workflows.workflow import GenerationInput, GenerationWorkflow
from fastapi import (
    Depends,
    FastAPI,
    HTTPException,
    Request,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel
from temporalio.client import Client

from api_gateway import deps
from api_gateway.deps import ModelRegistry, RedisLike

# Sample uploads are capped well below typical request-body limits so a single
# tenant can't exhaust gateway memory profiling an enormous file synchronously.
_MAX_SAMPLE_BYTES = 25 * 1024 * 1024


class RegisterModelRequest(BaseModel):
    name: str
    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    params: dict[str, object] = {}


class CreateDatasetRequest(BaseModel):
    name: str
    description: str = ""
    target_rows: int = 0
    # "description" (default, C0) proposes a schema via the LLM; "sample" creates a
    # draft with no schema yet -- POST /datasets/{id}/sample populates it.
    source: str = "description"


class UpdateDatasetRequest(BaseModel):
    name: str | None = None
    target_rows: int | None = None
    fields: list[FieldSpec] | None = None
    directives: dict[str, object] | None = None


class GenerateRequest(BaseModel):
    seed: int = 0


class CreateFromTemplateRequest(BaseModel):
    template_key: str
    name: str | None = None
    target_rows: int | None = None
    directives: dict[str, object] | None = None


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Anodyne API Gateway")

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        # Bind a request_id for correlation before auth runs; tenant_id is
        # rebound to the real value once get_tenant_context resolves the
        # token (see deps.get_tenant_context). "anonymous" covers requests
        # that never reach/pass auth (e.g. /healthz, a 401).
        request_id = str(uuid4())
        request.state.request_id = request_id
        bind_request_context(tenant_id="anonymous", request_id=request_id)
        return await call_next(request)

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/me")
    async def me(ctx: TenantContext = Depends(deps.get_tenant_context)) -> dict[str, object]:
        return ctx.model_dump(mode="json")

    @app.post("/models", status_code=201)
    async def register_model(
        body: RegisterModelRequest,
        ctx: TenantContext = Depends(deps.require("models:write")),
        registry: ModelRegistry = Depends(deps.get_model_registry),
    ) -> dict[str, object]:
        cfg = await registry.create(
            ctx.tenant_id,
            name=body.name,
            provider=body.provider,
            model=body.model,
            api_key=body.api_key,
            api_base=body.api_base,
            params=body.params,
        )
        data = cfg.model_dump(mode="json")
        data.pop("secret_ref", None)  # never expose refs
        return data

    @app.get("/models")
    async def list_models(
        ctx: TenantContext = Depends(deps.require("models:read")),
        registry: ModelRegistry = Depends(deps.get_model_registry),
    ) -> list[dict[str, object]]:
        out = []
        for cfg in await registry.list(ctx.tenant_id):
            d = cfg.model_dump(mode="json")
            d.pop("secret_ref", None)
            out.append(d)
        return out

    @app.delete("/models/{config_id}", status_code=204)
    async def delete_model(
        config_id: UUID,
        ctx: TenantContext = Depends(deps.require("models:delete")),
        registry: ModelRegistry = Depends(deps.get_model_registry),
    ) -> None:
        await registry.delete(ctx.tenant_id, config_id)

    @app.post("/llm/invoke")
    async def invoke(
        request: LLMRequest,
        ctx: TenantContext = Depends(deps.require("llm:invoke")),
        provider: LLMProvider = Depends(deps.get_llm_provider),
        registry: ModelRegistry = Depends(deps.get_model_registry),
    ) -> dict[str, object]:
        cfg = await registry.get(ctx.tenant_id, request.model_config_id)
        if cfg is None:
            raise HTTPException(404, "model config not found")
        resp = await provider.complete(cfg, request)
        return resp.model_dump(mode="json")

    @app.post("/datasets", status_code=201)
    async def create_dataset(
        body: CreateDatasetRequest,
        ctx: TenantContext = Depends(deps.require("datasets:write")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
        proposer: SchemaProposer = Depends(deps.get_schema_proposer),
    ) -> dict[str, object]:
        if body.source == "sample":
            # No description to propose a schema from -- the schema comes from
            # POST /datasets/{id}/sample once a sample is uploaded and profiled.
            spec = DatasetSpec(
                id=uuid4(),
                tenant_id=ctx.tenant_id,
                name=body.name,
                description=body.description,
                modality=Modality.TABULAR,
                source="sample",
                fields=[],
                target_rows=body.target_rows,
            )
            await repo.create_spec(spec)
            return spec.model_dump(mode="json")

        try:
            fields = await proposer.propose(body.description)
        except SchemaProposalError as exc:
            # Malformed LLM output is a client-fixable input problem (retry
            # with a clearer description), not a server fault -- was a bare
            # 500 before this fix.
            raise HTTPException(400, f"could not propose a schema: {exc}") from exc
        if not fields:
            raise HTTPException(
                400, "proposed schema has no fields; provide a more specific description"
            )
        spec = DatasetSpec(
            id=uuid4(),
            tenant_id=ctx.tenant_id,
            name=body.name,
            description=body.description,
            modality=Modality.TABULAR,
            source="description",
            fields=fields,
            target_rows=body.target_rows,
        )
        await repo.create_spec(spec)
        return spec.model_dump(mode="json")

    @app.post("/datasets/{dataset_id}/sample")
    async def upload_sample(
        dataset_id: UUID,
        file: UploadFile,
        ctx: TenantContext = Depends(deps.require("datasets:write")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
        profile_repo: ProfileRepository = Depends(deps.get_profile_repo),
        object_store: ObjectStore = Depends(deps.get_object_store),
        profiler: SampleProfiler = Depends(deps.get_sample_profiler),
    ) -> dict[str, object]:
        spec = await repo.get_spec(ctx.tenant_id, dataset_id)
        if spec is None:
            raise HTTPException(404, "dataset not found")
        if spec.source != "sample":
            raise HTTPException(
                400,
                "dataset source is not 'sample'; create the dataset with "
                'source="sample" to upload a sample',
            )
        data = await file.read()
        if len(data) > _MAX_SAMPLE_BYTES:
            raise HTTPException(413, "sample file too large")
        if not data:
            raise HTTPException(400, "sample file is empty")
        filename = file.filename or "sample.csv"
        key = f"datasets/{dataset_id}/sample/{filename}"
        await object_store.put(key, data)
        try:
            profile = profiler.profile(ctx.tenant_id, dataset_id, key, data, filename)
        except UnsupportedSampleFormatError as exc:
            raise HTTPException(400, str(exc)) from exc
        await profile_repo.save_profile(profile)
        spec.fields = fields_from_profile(profile)
        if spec.target_rows <= 0:
            spec.target_rows = profile.row_count
        await repo.update_spec(spec)
        return {"dataset": spec.model_dump(mode="json"), "profile": profile.model_dump(mode="json")}

    @app.get("/datasets")
    async def list_datasets(
        ctx: TenantContext = Depends(deps.require("datasets:read")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
    ) -> list[dict[str, object]]:
        return [s.model_dump(mode="json") for s in await repo.list_specs(ctx.tenant_id)]

    @app.get("/datasets/{dataset_id}")
    async def get_dataset(
        dataset_id: UUID,
        ctx: TenantContext = Depends(deps.require("datasets:read")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
    ) -> dict[str, object]:
        spec = await repo.get_spec(ctx.tenant_id, dataset_id)
        if spec is None:
            raise HTTPException(404, "dataset not found")
        return spec.model_dump(mode="json")

    @app.patch("/datasets/{dataset_id}")
    async def update_dataset(
        dataset_id: UUID,
        body: UpdateDatasetRequest,
        ctx: TenantContext = Depends(deps.require("datasets:write")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
    ) -> dict[str, object]:
        spec = await repo.get_spec(ctx.tenant_id, dataset_id)
        if spec is None:
            raise HTTPException(404, "dataset not found")
        if body.name is not None:
            spec.name = body.name
        if body.target_rows is not None:
            spec.target_rows = body.target_rows
        if body.fields is not None:
            spec.fields = body.fields
        if body.directives is not None:
            spec.directives = body.directives
        await repo.update_spec(spec)
        return spec.model_dump(mode="json")

    @app.get("/templates")
    async def get_templates(
        ctx: TenantContext = Depends(deps.require("datasets:read")),
    ) -> list[dict[str, object]]:
        return [t.model_dump(mode="json") for t in list_templates()]

    @app.post("/datasets/from-template", status_code=201)
    async def create_dataset_from_template(
        body: CreateFromTemplateRequest,
        ctx: TenantContext = Depends(deps.require("datasets:write")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
    ) -> dict[str, object]:
        template = get_template(body.template_key)
        if template is None:
            raise HTTPException(404, f"unknown template: {body.template_key}")
        spec = build_dataset_spec(
            template,
            tenant_id=ctx.tenant_id,
            name=body.name,
            target_rows=body.target_rows,
            directives=body.directives,
        )
        await repo.create_spec(spec)
        return spec.model_dump(mode="json")

    @app.post("/datasets/{dataset_id}/generate", status_code=202)
    async def start_generation(
        dataset_id: UUID,
        body: GenerateRequest,
        ctx: TenantContext = Depends(deps.require("datasets:write")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
        client: Client = Depends(deps.get_temporal_client),
    ) -> dict[str, object]:
        spec = await repo.get_spec(ctx.tenant_id, dataset_id)
        if spec is None:
            raise HTTPException(404, "dataset not found")
        if not spec.fields:
            # Catches the case a PATCH edited the schema down to zero fields
            # after creation (creation itself already rejects an empty
            # proposed schema -- see `create_dataset`).
            raise HTTPException(400, "dataset has no fields; edit the schema before generating")
        job = GenerationJob(id=uuid4(), tenant_id=ctx.tenant_id, dataset_id=dataset_id)
        # C0 does schema review *before* generate is called (the UI reviews
        # the proposed schema, then calls this route), so there is no
        # separate human step left to gate on here. Start the workflow
        # already-approved via signal-with-start so it doesn't park at
        # `awaiting_review` forever waiting for a signal nothing sends. The
        # workflow's HITL gate itself stays intact for when real pre-generate
        # review lands.
        # Only meaningful for source="sample" (the from-description path always uses
        # TabularSampler regardless); defaults to the permissive copula generator.
        method = str(spec.directives.get("synthesizer", "copula"))
        handle = await client.start_workflow(
            GenerationWorkflow.run,
            GenerationInput(
                job_id=str(job.id),
                dataset_id=str(dataset_id),
                tenant_id=str(ctx.tenant_id),
                target_rows=spec.target_rows,
                seed=body.seed,
                method=method,
            ),
            id=f"gen-{job.id}",
            task_queue="generation",
            start_signal="approve_schema",
        )
        job.workflow_id = handle.id
        await repo.save_job(job)
        return job.model_dump(mode="json")

    @app.get("/jobs/{job_id}")
    async def get_job_status(
        job_id: UUID,
        ctx: TenantContext = Depends(deps.require("datasets:read")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
    ) -> dict[str, object]:
        job = await repo.get_job(ctx.tenant_id, job_id)
        if job is None:
            raise HTTPException(404, "job not found")
        return job.model_dump(mode="json")

    @app.websocket("/jobs/{job_id}/stream")
    async def job_progress_stream(
        websocket: WebSocket,
        job_id: UUID,
        ctx: TenantContext = Depends(deps.require("datasets:read")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
        redis_client: RedisLike = Depends(deps.get_redis),
    ) -> None:
        await websocket.accept()
        # `require("datasets:read")` only checks the caller's role -- it says
        # nothing about whether THIS job belongs to their tenant (unlike
        # `GET /jobs/{id}`, which 404s via `repo.get_job(ctx.tenant_id, ...)`).
        # Without this check, any tenant with read access could stream any
        # other tenant's job progress by guessing/enumerating job ids.
        job = await repo.get_job(ctx.tenant_id, job_id)
        if job is None:
            await websocket.close(code=4404, reason="job not found")
            return
        channel = f"job:{job_id}"
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)

        # Forwarding messages never blocks on receiving from the client, so we
        # need a concurrent watcher to notice the client going away (otherwise
        # the loop below would spin forever after a disconnect).
        async def _watch_disconnect() -> None:
            with contextlib.suppress(WebSocketDisconnect):
                while True:
                    await websocket.receive()

        watcher = asyncio.ensure_future(_watch_disconnect())
        try:
            while not watcher.done():
                message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                if message is None:
                    await asyncio.sleep(0.01)
                    continue
                data = message["data"]
                if isinstance(data, bytes):
                    data = data.decode()
                await websocket.send_text(data)
        except WebSocketDisconnect:
            pass
        finally:
            watcher.cancel()
            with contextlib.suppress(Exception):
                await watcher
            await pubsub.unsubscribe(channel)
            await pubsub.close()

    @app.get("/datasets/{dataset_id}/versions")
    async def list_versions(
        dataset_id: UUID,
        ctx: TenantContext = Depends(deps.require("datasets:read")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
    ) -> list[dict[str, object]]:
        versions = await repo.list_versions(ctx.tenant_id, dataset_id)
        return [v.model_dump(mode="json") for v in versions]

    @app.get("/datasets/{dataset_id}/versions/{version_id}/download")
    async def download_version(
        dataset_id: UUID,
        version_id: UUID,
        ctx: TenantContext = Depends(deps.require("datasets:read")),
        repo: DatasetRepository = Depends(deps.get_dataset_repo),
        object_store: ObjectStore = Depends(deps.get_object_store),
    ) -> dict[str, str]:
        versions = await repo.list_versions(ctx.tenant_id, dataset_id)
        version = next((v for v in versions if v.id == version_id), None)
        if version is None:
            raise HTTPException(404, "version not found")
        url = await object_store.presigned_url(version.artifact_uri)
        return {"url": url}

    return app
