from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANODYNE_", env_file=".env")
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


def get_settings() -> Settings:
    return Settings()
