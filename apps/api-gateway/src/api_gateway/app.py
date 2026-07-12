from __future__ import annotations

from anodyne_core.models import LLMRequest, TenantContext
from anodyne_core.ports import LLMProvider
from anodyne_observability.logging import configure_logging
from fastapi import Depends, FastAPI, HTTPException

from api_gateway import deps
from api_gateway.deps import ModelRegistry


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Anodyne API Gateway")

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz() -> dict[str, str]:
        return {"status": "ready"}

    @app.get("/me")
    async def me(ctx: TenantContext = Depends(deps.get_tenant_context)) -> dict[str, object]:
        return ctx.model_dump(mode="json")

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

    return app
