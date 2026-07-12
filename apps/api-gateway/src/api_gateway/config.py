from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANODYNE_", env_file=".env")
    database_url: str = "postgresql+asyncpg://app:app@localhost:5432/anodyne"
    oidc_issuer: str = "http://localhost:8080/realms/anodyne"
    oidc_jwks_url: str = "http://localhost:8080/realms/anodyne/protocol/openid-connect/certs"
    oidc_audience: str = "anodyne"
    secret_key: str = ""  # base64 Fernet key; required in prod
    s3_bucket: str = "anodyne"


def get_settings() -> Settings:
    return Settings()
