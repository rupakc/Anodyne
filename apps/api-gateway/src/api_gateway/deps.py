from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache
from typing import Protocol
from uuid import UUID

import jwt
from anodyne_core.models import ModelConfig, TenantContext
from anodyne_core.ports import AuthorizationPolicy, LLMProvider, SecretStore
from anodyne_llm.adapter import LiteLLMProvider
from anodyne_llm.registry import SqlModelRegistry
from anodyne_storage.db import make_engine
from anodyne_storage.secrets import FernetSecretStore
from anodyne_tenancy.authz import RoleBasedPolicy
from anodyne_tenancy.oidc import AuthError, TokenValidator
from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine

from api_gateway.config import Settings, get_settings


class SecretStoreConfigError(RuntimeError):
    """Raised when `ANODYNE_SECRET_KEY` is missing or not a valid Fernet key."""


class ModelRegistry(Protocol):
    """Structural type for the model-config registry consumed by the gateway.

    `SqlModelRegistry` (anodyne_llm.registry) is the real, DB-backed
    implementation; tests substitute in-memory fakes via
    `app.dependency_overrides[get_model_registry]`.
    """

    async def create(
        self,
        tenant_id: UUID,
        *,
        name: str,
        provider: str,
        model: str,
        api_key: str | None,
        api_base: str | None,
        params: dict[str, object],
    ) -> ModelConfig: ...

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None: ...

    async def list(self, tenant_id: UUID) -> list[ModelConfig]: ...

    async def delete(self, tenant_id: UUID, config_id: UUID) -> None: ...


@lru_cache
def _validator(issuer: str, jwks_url: str, audience: str) -> TokenValidator:
    return TokenValidator(jwt.PyJWKClient(jwks_url), issuer=issuer, audience=audience)


def get_tenant_context(
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> TenantContext:
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        v = _validator(settings.oidc_issuer, settings.oidc_jwks_url, settings.oidc_audience)
        return v.validate(authorization.removeprefix("Bearer "))
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc


def get_policy() -> AuthorizationPolicy:
    return RoleBasedPolicy()


def require(permission: str) -> Callable[..., TenantContext]:
    def _dep(
        ctx: TenantContext = Depends(get_tenant_context),
        policy: AuthorizationPolicy = Depends(get_policy),
    ) -> TenantContext:
        if not policy.is_permitted(ctx, permission):
            raise HTTPException(403, f"missing permission: {permission}")
        return ctx

    return _dep


@lru_cache
def _engine(database_url: str) -> AsyncEngine:
    return make_engine(database_url)


@lru_cache
def _secret_store(secret_key: str) -> SecretStore:
    try:
        return FernetSecretStore(secret_key.encode())
    except ValueError as exc:
        raise SecretStoreConfigError(
            "ANODYNE_SECRET_KEY is missing or not a valid Fernet key. Generate one with: "
            'python -c "from cryptography.fernet import Fernet; '
            'print(Fernet.generate_key().decode())" and set it in your .env.'
        ) from exc


def get_llm_provider(settings: Settings = Depends(get_settings)) -> LLMProvider:
    """Real LiteLLM-backed provider: decrypts per-model secrets via the Fernet store.

    Overridden in tests via `app.dependency_overrides[get_llm_provider]`.
    """
    return LiteLLMProvider(_secret_store(settings.secret_key))


def get_model_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    """Real, DB-backed registry: SQL storage + Fernet-encrypted secret refs.

    Overridden in tests via `app.dependency_overrides[get_model_registry]`.
    """
    return SqlModelRegistry(_engine(settings.database_url), _secret_store(settings.secret_key))
