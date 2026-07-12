from __future__ import annotations

import asyncio
from collections.abc import Callable
from functools import lru_cache
from typing import Any, Protocol, cast
from uuid import UUID

import jwt
import redis.asyncio as redis
from anodyne_audio.registry import SqlAudioProviderRegistry
from anodyne_core.models import ModelConfig, TenantContext
from anodyne_core.ports import AuthorizationPolicy, LLMProvider, ObjectStore, SecretStore
from anodyne_dataset.ports import (
    DatasetRepository,
    ProfileRepository,
    SampleProfiler,
    SchemaProposer,
)
from anodyne_generation.proposer import LLMSchemaProposer
from anodyne_image.registry import SqlImageProviderRegistry
from anodyne_llm.adapter import LiteLLMProvider
from anodyne_llm.registry import SqlModelRegistry
from anodyne_observability.logging import bind_request_context
from anodyne_storage.dataset_repo import SqlDatasetRepository
from anodyne_storage.db import make_engine
from anodyne_storage.objectstore import S3ObjectStore
from anodyne_storage.secrets import FernetSecretStore
from anodyne_tabular.profiler import PandasSampleProfiler
from anodyne_tenancy.authz import RoleBasedPolicy
from anodyne_tenancy.oidc import AuthError, TokenValidator
from anodyne_video.ports import VideoProviderRegistry
from anodyne_video.registry import SqlVideoProviderRegistry
from fastapi import Depends, Header, HTTPException
from sqlalchemy.ext.asyncio import AsyncEngine
from starlette.requests import HTTPConnection
from temporalio.client import Client

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
    request: HTTPConnection,
    authorization: str = Header(default=""),
    settings: Settings = Depends(get_settings),
) -> TenantContext:
    # `HTTPConnection` (not `Request`) so this same dependency — and anything
    # built on it, e.g. `require(...)` — works for both HTTP routes and the
    # `WS /jobs/{id}/stream` route: FastAPI cannot inject a `Request` into a
    # websocket handler, but `Request`/`WebSocket` both subclass `HTTPConnection`.
    if not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    try:
        v = _validator(settings.oidc_issuer, settings.oidc_jwks_url, settings.oidc_audience)
        ctx = v.validate(authorization.removeprefix("Bearer "))
    except AuthError as exc:
        raise HTTPException(401, str(exc)) from exc
    # Rebind the request-context request_id (set by the app's HTTP
    # middleware) together with the now-known real tenant_id, so subsequent
    # log lines in this request correlate to the authenticated tenant.
    request_id = getattr(request.state, "request_id", "unknown")
    bind_request_context(tenant_id=str(ctx.tenant_id), request_id=request_id)
    return ctx


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


def get_video_provider_registry(
    settings: Settings = Depends(get_settings),
) -> VideoProviderRegistry:
    """Real, DB-backed `VideoProviderRegistry` (tenant video-provider configs).

    Overridden in tests via `app.dependency_overrides[get_video_provider_registry]`.
    """
    engine = _engine(settings.database_url)
    return SqlVideoProviderRegistry(engine, _secret_store(settings.secret_key))


def get_dataset_repo(settings: Settings = Depends(get_settings)) -> DatasetRepository:
    """Real, DB-backed `DatasetRepository` (specs, generation jobs, versions).

    Overridden in tests via `app.dependency_overrides[get_dataset_repo]`.
    """
    return SqlDatasetRepository(_engine(settings.database_url))


def get_profile_repo(settings: Settings = Depends(get_settings)) -> ProfileRepository:
    """Real, DB-backed `ProfileRepository` for uploaded-sample profiles.

    `SqlDatasetRepository` implements both `DatasetRepository` and `ProfileRepository`.
    Overridden in tests via `app.dependency_overrides[get_profile_repo]`.
    """
    return SqlDatasetRepository(_engine(settings.database_url))


def get_sample_profiler() -> SampleProfiler:
    """Real, pandas-based `SampleProfiler`.

    Overridden in tests via `app.dependency_overrides[get_sample_profiler]`.
    """
    return PandasSampleProfiler()


def get_image_provider_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    """Real, DB-backed per-tenant image-provider registry (separate table from
    the LLM `model_configs` one -- see `anodyne_image`'s migration docstring).

    Shares `ModelRegistry`'s structural type (identical CRUD shape over
    `ModelConfig`); overridden in tests via
    `app.dependency_overrides[get_image_provider_registry]`.
    """
    return SqlImageProviderRegistry(
        _engine(settings.database_url), _secret_store(settings.secret_key)
    )


def get_audio_provider_registry(settings: Settings = Depends(get_settings)) -> ModelRegistry:
    """Real, DB-backed per-tenant audio-provider registry (separate table from
    the LLM `model_configs` one, mirroring image/video).

    Shares `ModelRegistry`'s structural type (identical CRUD shape over
    `ModelConfig`); overridden in tests via
    `app.dependency_overrides[get_audio_provider_registry]`.
    """
    return SqlAudioProviderRegistry(
        _engine(settings.database_url), _secret_store(settings.secret_key)
    )


async def get_schema_proposer(
    ctx: TenantContext = Depends(get_tenant_context),
    provider: LLMProvider = Depends(get_llm_provider),
    registry: ModelRegistry = Depends(get_model_registry),
) -> SchemaProposer:
    """Real, LLM-backed `SchemaProposer`.

    Uses the tenant's first configured `ModelConfig` to drive the proposal.
    (A dedicated "default model for schema proposal" flag is a reasonable
    follow-up if tenants need to pick a specific model for this.)
    Overridden in tests via `app.dependency_overrides[get_schema_proposer]`.
    """
    configs = await registry.list(ctx.tenant_id)
    if not configs:
        raise HTTPException(400, "no model configured for this tenant; register one first")
    return LLMSchemaProposer(provider, configs[0])


_temporal_client: Client | None = None
_temporal_client_lock = asyncio.Lock()


async def get_temporal_client(settings: Settings = Depends(get_settings)) -> Client:
    """Real Temporal `Client`, connected once and cached for the process lifetime.

    Overridden in tests via `app.dependency_overrides[get_temporal_client]`
    (no live Temporal server needed for non-integration tests).
    """
    global _temporal_client
    if _temporal_client is None:
        async with _temporal_client_lock:
            if _temporal_client is None:
                _temporal_client = await Client.connect(settings.temporal_address)
    return _temporal_client


class PubSubLike(Protocol):
    """Structural type for the pub/sub handle used by `WS /jobs/{id}/stream`."""

    async def subscribe(self, *channels: str) -> None: ...

    # Mirrors redis-py's `PubSub.get_message` signature (its `timeout` isn't
    # an `asyncio.timeout`, hence the noqa).
    async def get_message(
        self,
        *,
        ignore_subscribe_messages: bool = ...,
        timeout: float | None = ...,  # noqa: ASYNC109
    ) -> dict[str, Any] | None: ...

    async def unsubscribe(self, *channels: str) -> None: ...

    async def close(self) -> None: ...


class RedisLike(Protocol):
    """Structural type for the Redis client consumed by the gateway.

    `redis.asyncio.Redis` is the real implementation; tests substitute
    in-memory fakes via `app.dependency_overrides[get_redis]`.
    """

    def pubsub(self) -> PubSubLike: ...


@lru_cache
def _redis(redis_url: str) -> RedisLike:
    # `Redis.pubsub()` takes `**kwargs` and returns the concrete `PubSub`
    # class; both are structurally compatible with `RedisLike`/`PubSubLike`
    # at the call sites we use, so the cast is safe.
    return cast(RedisLike, redis.from_url(redis_url))


def get_redis(settings: Settings = Depends(get_settings)) -> RedisLike:
    """Real Redis client, used to subscribe to `job:{id}` progress channels.

    Overridden in tests via `app.dependency_overrides[get_redis]`.
    """
    return _redis(settings.redis_url)


@lru_cache
def _s3_client() -> Any:
    import boto3  # type: ignore[import-untyped]

    return boto3.client("s3")


def get_object_store(
    ctx: TenantContext = Depends(get_tenant_context),
    settings: Settings = Depends(get_settings),
) -> ObjectStore:
    """Real, tenant-scoped `ObjectStore` backed by S3/MinIO.

    Overridden in tests via `app.dependency_overrides[get_object_store]`.
    """
    return S3ObjectStore(settings.s3_bucket, ctx.tenant_id, client=_s3_client())
