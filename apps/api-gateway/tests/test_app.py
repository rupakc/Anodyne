from uuid import uuid4

import pytest
from anodyne_core.models import LLMResponse, ModelConfig, Role, TenantContext, Usage, User
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role = Role.MEMBER) -> TenantContext:
    tid = uuid4()
    u = User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tid, user=u, roles=[role])


@pytest.fixture
def client_and_app() -> tuple[AsyncClient, object]:
    app = create_app()
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t"), app


async def test_healthz_is_public(client_and_app):  # type: ignore[no-untyped-def]
    client, _ = client_and_app
    assert (await client.get("/healthz")).status_code == 200


async def test_me_returns_context(client_and_app):  # type: ignore[no-untyped-def]
    client, app = client_and_app
    ctx = _ctx()
    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    r = await client.get("/me")
    assert r.status_code == 200
    assert r.json()["user"]["email"] == "u@x.io"


async def test_invoke_requires_permission(client_and_app):  # type: ignore[no-untyped-def]
    client, app = client_and_app
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER)  # no llm:invoke
    r = await client.post(
        "/llm/invoke",
        json={"model_config_id": str(uuid4()), "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.status_code == 403


async def test_invoke_calls_provider(client_and_app):  # type: ignore[no-untyped-def]
    client, app = client_and_app
    ctx = _ctx(Role.MEMBER)
    cfg = ModelConfig(
        id=uuid4(),
        tenant_id=ctx.tenant_id,
        name="c",
        provider="openai",
        model="gpt-4o",
        secret_ref="ref",
    )

    class _Provider:
        async def complete(self, config, request):  # type: ignore[no-untyped-def]
            return LLMResponse(content="pong", usage=Usage(total_tokens=2), cost=0.0)

        def stream(self, config, request): ...  # type: ignore[no-untyped-def]

    class _Registry:
        async def get(self, tenant_id, config_id):  # type: ignore[no-untyped-def]
            return cfg

    app.dependency_overrides[deps.get_tenant_context] = lambda: ctx
    app.dependency_overrides[deps.get_llm_provider] = lambda: _Provider()
    app.dependency_overrides[deps.get_model_registry] = lambda: _Registry()

    r = await client.post(
        "/llm/invoke",
        json={"model_config_id": str(cfg.id), "messages": [{"role": "user", "content": "ping"}]},
    )
    assert r.status_code == 200
    assert r.json()["content"] == "pong"
