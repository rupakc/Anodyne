from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # `extra="ignore"`: tolerate a shared `.env` that also holds sibling-service
    # vars (AWS_* for boto3, LLM keys) — the dotenv source would otherwise reject
    # unmapped keys as extra_forbidden and crash startup.
    model_config = SettingsConfigDict(env_prefix="ANODYNE_", env_file=".env", extra="ignore")
    # Non-superuser runtime role (see api-gateway config): the app must never
    # connect as the migration superuser, which bypasses RLS.
    database_url: str = "postgresql+asyncpg://anodyne_app:anodyne_app@localhost:5432/anodyne"
    secret_key: str = ""  # base64 Fernet key; enables the qualitative LLM judge when set
    s3_bucket: str = "anodyne"
    temporal_address: str = "localhost:7233"
    redis_url: str = "redis://localhost:6379/0"
    ray_address: str = ""


def get_settings() -> Settings:
    return Settings()
