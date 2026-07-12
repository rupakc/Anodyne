from __future__ import annotations

from typing import Any, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

import jwt
from anodyne_core.models import Role, TenantContext, User


class AuthError(Exception): ...


class _JWKSClient(Protocol):
    def get_signing_key_from_jwt(self, token: str) -> Any: ...


class TokenValidator:
    def __init__(self, jwks_client: _JWKSClient, issuer: str, audience: str) -> None:
        self._jwks = jwks_client
        self._issuer = issuer
        self._audience = audience

    def validate(self, token: str) -> TenantContext:
        try:
            signing_key = self._jwks.get_signing_key_from_jwt(token)
            claims = jwt.decode(
                token,
                signing_key.key,
                algorithms=["RS256"],
                audience=self._audience,
                issuer=self._issuer,
            )
        except jwt.PyJWTError as exc:
            raise AuthError(str(exc)) from exc

        tenant_raw = claims.get("org_id") or claims.get("tenant_id")
        if not tenant_raw:
            raise AuthError("token has no tenant/org claim")
        tenant_id = UUID(str(tenant_raw))
        roles = [
            Role(r)
            for r in claims.get("realm_access", {}).get("roles", [])
            if r in Role._value2member_map_
        ]
        subject = str(claims["sub"])
        user = User(
            id=uuid5(NAMESPACE_URL, subject),
            tenant_id=tenant_id,
            subject=subject,
            email=str(claims.get("email", "")),
            roles=roles,
        )
        return TenantContext(tenant_id=tenant_id, user=user, roles=roles)
