from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ANODYNE_", env_file=".env")

    temporal_address: str = "localhost:7233"
    # Ray client address. Defaults to the `ray-head` container from `make up`
    # so `make dev` runs distributed out of the box. Set to "" to fall back to
    # an embedded local Ray instance — see `anodyne_compute.ray_init`.
    ray_address: str = "ray://localhost:10001"
    redis_url: str = "redis://localhost:6379/0"
    # Non-superuser runtime role (`anodyne_app`), matching api-gateway's
    # Settings — see .env.example and docs/dev-runbook.md.
    database_url: str = "postgresql+asyncpg://anodyne_app:anodyne_app@localhost:5432/anodyne"
    s3_bucket: str = "anodyne"
    secret_key: str = ""  # base64 Fernet key; required in prod
    # From-sample tabular synthesis (see anodyne_tabular). CTGAN/TVAE epochs are kept
    # small by default -- tune upward for higher fidelity at the cost of fit time.
    tabular_ctgan_epochs: int = 100
    # SDV is BSL 1.1 (separately licensed) -- opt-in per architecture decision; a tenant
    # must also request `directives["synthesizer"] = "sdv"` for it to actually be used.
    tabular_enable_sdv: bool = False


def get_settings() -> Settings:
    return Settings()
