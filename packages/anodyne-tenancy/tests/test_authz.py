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
