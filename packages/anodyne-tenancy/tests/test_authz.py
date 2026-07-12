from uuid import uuid4

from anodyne_core.models import Role, TenantContext, User
from anodyne_tenancy.authz import RoleBasedPolicy


def _ctx(role: Role) -> TenantContext:
    u = User(id=uuid4(), tenant_id=uuid4(), subject="s", email="a@b.c", roles=[role])
    return TenantContext(tenant_id=u.tenant_id, user=u, roles=[role])


def test_owner_can_delete_models() -> None:
    assert RoleBasedPolicy().is_permitted(_ctx(Role.OWNER), "models:delete")


def test_viewer_cannot_write_models() -> None:
    p = RoleBasedPolicy()
    assert p.is_permitted(_ctx(Role.VIEWER), "models:read")
    assert not p.is_permitted(_ctx(Role.VIEWER), "models:write")


def test_member_can_invoke_llm() -> None:
    assert RoleBasedPolicy().is_permitted(_ctx(Role.MEMBER), "llm:invoke")


def test_viewer_can_read_but_not_write_datasets() -> None:
    p = RoleBasedPolicy()
    assert p.is_permitted(_ctx(Role.VIEWER), "datasets:read")
    assert not p.is_permitted(_ctx(Role.VIEWER), "datasets:write")


def test_member_can_write_datasets() -> None:
    assert RoleBasedPolicy().is_permitted(_ctx(Role.MEMBER), "datasets:write")


def test_viewer_can_read_but_not_write_image_providers() -> None:
    p = RoleBasedPolicy()
    assert p.is_permitted(_ctx(Role.VIEWER), "image_providers:read")
    assert not p.is_permitted(_ctx(Role.VIEWER), "image_providers:write")


def test_member_can_write_but_not_delete_image_providers() -> None:
    p = RoleBasedPolicy()
    assert p.is_permitted(_ctx(Role.MEMBER), "image_providers:write")
    assert not p.is_permitted(_ctx(Role.MEMBER), "image_providers:delete")


def test_admin_can_delete_image_providers() -> None:
    assert RoleBasedPolicy().is_permitted(_ctx(Role.ADMIN), "image_providers:delete")


def test_viewer_can_read_but_not_write_video_providers() -> None:
    p = RoleBasedPolicy()
    assert p.is_permitted(_ctx(Role.VIEWER), "video_providers:read")
    assert not p.is_permitted(_ctx(Role.VIEWER), "video_providers:write")


def test_member_can_write_but_not_delete_video_providers() -> None:
    p = RoleBasedPolicy()
    assert p.is_permitted(_ctx(Role.MEMBER), "video_providers:write")
    assert not p.is_permitted(_ctx(Role.MEMBER), "video_providers:delete")


def test_admin_can_delete_video_providers() -> None:
    assert RoleBasedPolicy().is_permitted(_ctx(Role.ADMIN), "video_providers:delete")
