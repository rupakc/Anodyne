from uuid import uuid4

import pytest
from anodyne_core.models import ModelConfig, Role, TenantContext, User
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role) -> TenantContext:
    tid = uuid4()
    u = User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tid, user=u, roles=[role])


class _FakeRegistry:
    def __init__(self) -> None:
        self.store: dict[object, ModelConfig] = {}

    async def create(  # type: ignore[no-untyped-def]
        self, tenant_id, *, name, provider, model, api_key, api_base, params
    ):
        cfg = ModelConfig(
            id=uuid4(),
            tenant_id=tenant_id,
            name=name,
            provider=provider,
            model=model,
            params=params,
            secret_ref="ref" if api_key else None,
            api_base=api_base,
        )
        self.store[cfg.id] = cfg
        return cfg

    async def list(self, tenant_id):  # type: ignore[no-untyped-def]
        return list(self.store.values())

    async def delete(self, tenant_id, config_id):  # type: ignore[no-untyped-def]
        self.store.pop(config_id, None)


@pytest.fixture
def wired() -> tuple[AsyncClient, object]:
    app = create_app()
    reg = _FakeRegistry()
    app.dependency_overrides[deps.get_model_registry] = lambda: reg
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t"), app


async def test_register_hides_secret_ref(wired):  # type: ignore[no-untyped-def]
    client, app = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER)
    r = await client.post(
        "/models", json={"name": "c", "provider": "openai", "model": "gpt-4o", "api_key": "sk-x"}
    )
    assert r.status_code == 201
    assert "secret_ref" not in r.json()


async def test_viewer_cannot_register(wired):  # type: ignore[no-untyped-def]
    client, app = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER)
    r = await client.post("/models", json={"name": "c", "provider": "openai", "model": "gpt-4o"})
    assert r.status_code == 403
