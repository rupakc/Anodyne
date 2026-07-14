from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `extra="ignore"`: a shared `.env` (see scripts/setup-local-secrets.sh) also
    # carries vars for sibling services (AWS_* for boto3, LLM keys, worker-only
    # settings). The dotenv source would otherwise reject those as extra_forbidden
    # and crash startup; each service simply ignores keys that aren't its fields.
    model_config = SettingsConfigDict(env_prefix="ANODYNE_", env_file=".env", extra="ignore")
    # Non-superuser runtime role (`anodyne_app`), NOT the `postgres` migration
    # superuser — see .env.example and docs/dev-runbook.md. Superusers bypass
    # row-level security even with FORCE ROW LEVEL SECURITY, so the app must
    # never connect as `postgres`.
    database_url: str = "postgresql+asyncpg://anodyne_app:anodyne_app@localhost:5432/anodyne"
    oidc_issuer: str = "http://localhost:8080/realms/anodyne"
    oidc_jwks_url: str = "http://localhost:8080/realms/anodyne/protocol/openid-connect/certs"
    oidc_audience: str = "anodyne"
    secret_key: str = ""  # base64 Fernet key; required in prod
    s3_bucket: str = "anodyne"
    temporal_address: str = "localhost:7233"
    redis_url: str = "redis://localhost:6379/0"
    # Comma-separated browser origins allowed to call the API (CORS). The web
    # app runs on a different origin than the gateway, so its origin must be
    # listed or the browser blocks every authenticated request as a CORS
    # failure ("Failed to fetch"). Defaults cover local dev; set explicitly
    # (e.g. the deployed web origin) in other environments.
    cors_allow_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    # TTL (seconds) applied to presigned download URLs the gateway hands back
    # for artifacts/reports. Default 24h so a link the user opens well after a
    # page was rendered doesn't fail with "Signature has expired". Env:
    # `ANODYNE_PRESIGNED_TTL`.
    presigned_ttl: int = 86400
    # Preferred provider for LLM tasks that don't pin an explicit
    # `model_config_id`: the tenant's first config whose `provider` matches
    # this (case-insensitive) is used; falls back to `configs[0]` if no such
    # config exists. Avoids silently defaulting to a slow local Ollama config
    # just because it happened to be registered first. Env:
    # `ANODYNE_DEFAULT_LLM_PROVIDER`.
    default_llm_provider: str = "gemini"


def get_settings() -> Settings:
    return Settings()
