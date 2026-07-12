from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANODYNE_", env_file=".env")

    temporal_address: str = "localhost:7233"
    # Ray cluster address, e.g. "ray://host:10001". Falsy ("") starts/uses a
    # local Ray instance instead — see `anodyne_compute.ray_init`.
    ray_address: str = ""
    redis_url: str = "redis://localhost:6379/0"
    # Non-superuser runtime role (`anodyne_app`), matching api-gateway's
    # Settings — see .env.example and docs/dev-runbook.md.
    database_url: str = "postgresql+asyncpg://anodyne_app:anodyne_app@localhost:5432/anodyne"
    s3_bucket: str = "anodyne"
    secret_key: str = ""  # base64 Fernet key; required in prod


def get_settings() -> Settings:
    return Settings()
