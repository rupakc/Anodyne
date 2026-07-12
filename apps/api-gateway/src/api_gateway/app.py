from __future__ import annotations

from uuid import UUID

from anodyne_core.models import LLMRequest, TenantContext
from anodyne_core.ports import LLMProvider
from anodyne_observability.logging import configure_logging
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

from api_gateway import deps
from api_gateway.deps import ModelRegistry


class RegisterModelRequest(BaseModel):
    name: str
    provider: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    params: dict[str, object] = {}


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

    return app
