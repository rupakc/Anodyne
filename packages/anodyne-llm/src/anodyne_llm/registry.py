from __future__ import annotations

from uuid import UUID, uuid4

from anodyne_core.models import ModelConfig
from anodyne_core.ports import SecretStore
from anodyne_storage.db import model_configs, tenant_session
from sqlalchemy import delete, insert, select
from sqlalchemy.ext.asyncio import AsyncEngine


def _row_to_config(row: object) -> ModelConfig:
    m = row._mapping  # type: ignore[attr-defined]
    return ModelConfig(
        id=m["id"],
        tenant_id=m["tenant_id"],
        name=m["name"],
        provider=m["provider"],
        model=m["model"],
        params=m["params"],
        secret_ref=m["secret_ref"],
        api_base=m["api_base"],
        enabled=str(m["enabled"]).lower() == "true",
    )


class SqlModelRegistry:
    def __init__(self, engine: AsyncEngine, secret_store: SecretStore) -> None:
        self._engine = engine
        self._secrets = secret_store

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
    ) -> ModelConfig:
        secret_ref = self._secrets.encrypt(api_key) if api_key else None
        cid = uuid4()
        async with tenant_session(self._engine, tenant_id) as s:
            await s.execute(
                insert(model_configs).values(
                    id=cid,
                    tenant_id=tenant_id,
                    name=name,
                    provider=provider,
                    model=model,
                    params=params,
                    secret_ref=secret_ref,
                    api_base=api_base,
                    enabled="true",
                )
            )
            await s.commit()
        return ModelConfig(
            id=cid,
            tenant_id=tenant_id,
            name=name,
            provider=provider,
            model=model,
            params=params,
            secret_ref=secret_ref,
            api_base=api_base,
        )

    async def get(self, tenant_id: UUID, config_id: UUID) -> ModelConfig | None:
        # Explicit tenant_id filter is defense-in-depth: Postgres RLS
        # (tenant_session's `app.tenant_id` GUC) is the primary isolation
        # boundary, but cross-tenant access must be impossible even if RLS is
        # ever misconfigured, disabled, or bypassed (e.g. a superuser DSN).
        async with tenant_session(self._engine, tenant_id) as s:
            row = (
                await s.execute(
                    select(model_configs).where(
                        model_configs.c.id == config_id,
                        model_configs.c.tenant_id == tenant_id,
                    )
                )
            ).first()
            return _row_to_config(row) if row else None

    async def list(self, tenant_id: UUID) -> list[ModelConfig]:
        async with tenant_session(self._engine, tenant_id) as s:
            rows = (
                await s.execute(select(model_configs).where(model_configs.c.tenant_id == tenant_id))
            ).all()
            return [_row_to_config(r) for r in rows]

    async def delete(self, tenant_id: UUID, config_id: UUID) -> None:
        async with tenant_session(self._engine, tenant_id) as s:
            await s.execute(
                delete(model_configs).where(
                    model_configs.c.id == config_id,
                    model_configs.c.tenant_id == tenant_id,
                )
            )
            await s.commit()
