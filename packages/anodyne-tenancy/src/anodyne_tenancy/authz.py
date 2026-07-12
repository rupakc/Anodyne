from __future__ import annotations

from anodyne_core.models import Role, TenantContext
from anodyne_core.ports import AuthorizationPolicy

_VIEWER = {
    "models:read",
    "datasets:read",
    "image_providers:read",
    "video_providers:read",
    "audio_providers:read",
    "perturbations:read",
    "evaluations:read",
    "annotations:read",
    "reviews:read",
}
_MEMBER = _VIEWER | {
    "llm:invoke",
    "models:write",
    "datasets:write",
    "image_providers:write",
    "video_providers:write",
    "audio_providers:write",
    "perturbations:write",
    "evaluations:write",
    "annotations:write",
    "reviews:write",
}
_ADMIN = _MEMBER | {
    "models:delete",
    "users:read",
    "image_providers:delete",
    "video_providers:delete",
    "audio_providers:delete",
}
_OWNER = _ADMIN | {"users:write", "tenant:admin"}

PERMISSIONS: dict[Role, set[str]] = {
    Role.VIEWER: _VIEWER,
    Role.MEMBER: _MEMBER,
    Role.ADMIN: _ADMIN,
    Role.OWNER: _OWNER,
}


class RoleBasedPolicy(AuthorizationPolicy):
    def is_permitted(self, ctx: TenantContext, permission: str) -> bool:
        return any(permission in PERMISSIONS.get(role, set()) for role in ctx.roles)
