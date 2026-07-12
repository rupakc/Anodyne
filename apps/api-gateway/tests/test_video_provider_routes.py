from uuid import UUID, uuid4

import pytest
from anodyne_core.models import Role, TenantContext, User
from anodyne_video.models import VideoProviderConfig
from api_gateway import deps
from api_gateway.app import create_app
from httpx import ASGITransport, AsyncClient


def _ctx(role: Role, tenant_id: UUID | None = None) -> TenantContext:
    tid = tenant_id or uuid4()
    u = User(id=uuid4(), tenant_id=tid, subject="s", email="u@x.io", roles=[role])
    return TenantContext(tenant_id=tid, user=u, roles=[role])


class _FakeVideoProviderRegistry:
    def __init__(self) -> None:
        self.store: dict[object, VideoProviderConfig] = {}

    async def create(  # type: ignore[no-untyped-def]
        self, tenant_id, *, name, provider, model, api_key, api_base, params
    ):
        cfg = VideoProviderConfig(
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
        return [c for c in self.store.values() if c.tenant_id == tenant_id]

    async def delete(self, tenant_id, config_id):  # type: ignore[no-untyped-def]
        self.store.pop(config_id, None)


@pytest.fixture
def wired() -> tuple[AsyncClient, object, _FakeVideoProviderRegistry]:
    app = create_app()
    reg = _FakeVideoProviderRegistry()
    app.dependency_overrides[deps.get_video_provider_registry] = lambda: reg
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://t"), app, reg


async def test_register_hides_secret_ref(wired):  # type: ignore[no-untyped-def]
    client, app, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER)
    r = await client.post(
        "/video-providers",
        json={
            "name": "c",
            "provider": "replicate",
            "model": "stability-ai/stable-video-diffusion",
            "api_key": "sk-x",
        },
    )
    assert r.status_code == 201
    assert "secret_ref" not in r.json()


async def test_viewer_cannot_register(wired):  # type: ignore[no-untyped-def]
    client, app, _ = wired
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER)
    r = await client.post(
        "/video-providers",
        json={"name": "c", "provider": "replicate", "model": "m"},
    )
    assert r.status_code == 403


async def test_list_hides_secrets_and_is_tenant_scoped(wired):  # type: ignore[no-untyped-def]
    client, app, reg = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    await client.post("/video-providers", json={"name": "c", "provider": "replicate", "model": "m"})

    r = await client.get("/video-providers")

    assert r.status_code == 200
    body = r.json()
    assert len(body) == 1
    assert "secret_ref" not in body[0]
    assert reg.store  # actually persisted


async def test_viewer_can_list_but_not_delete(wired):  # type: ignore[no-untyped-def]
    client, app, reg = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post(
        "/video-providers", json={"name": "c", "provider": "replicate", "model": "m"}
    )
    config_id = created.json()["id"]

    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.VIEWER, tid)
    r = await client.get("/video-providers")
    assert r.status_code == 200

    r = await client.delete(f"/video-providers/{config_id}")
    assert r.status_code == 403
    assert reg.store  # not deleted


async def test_admin_can_delete(wired):  # type: ignore[no-untyped-def]
    client, app, reg = wired
    tid = uuid4()
    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.MEMBER, tid)
    created = await client.post(
        "/video-providers", json={"name": "c", "provider": "replicate", "model": "m"}
    )
    config_id = created.json()["id"]

    app.dependency_overrides[deps.get_tenant_context] = lambda: _ctx(Role.ADMIN, tid)
    r = await client.delete(f"/video-providers/{config_id}")

    assert r.status_code == 204
    assert not reg.store
