"""Idempotent demo-tenant seed for local dev.

Invoked via `make seed` (`python -m api_gateway.seed`) after `make migrate`.
Upserts the tenant row that the demo Keycloak user (`demo@anodyne.dev`, see
`infra/docker/keycloak/anodyne-realm.json`) is scoped to via its `org_id`
token claim, so a freshly-migrated dev database has a tenant to attach model
configs and `/llm/invoke` requests to.
"""

from __future__ import annotations

import asyncio
from uuid import UUID

from anodyne_storage.db import tenant_session, tenants
from sqlalchemy.dialects.postgresql import insert as pg_insert

from api_gateway.config import get_settings
from api_gateway.deps import _engine, _secret_store

# Must match the `org_id` user attribute on `demo@anodyne.dev` in
# infra/docker/keycloak/anodyne-realm.json.
DEMO_TENANT_ID = UUID("11111111-1111-1111-1111-111111111111")
DEMO_TENANT_NAME = "Demo Tenant"
DEMO_ORG_REF = "demo-tenant"


async def seed() -> None:
    settings = get_settings()
    _secret_store(settings.secret_key)  # fail fast on a missing/invalid ANODYNE_SECRET_KEY
    engine = _engine(settings.database_url)
    try:
        # `tenants` carries a FORCE RLS policy keyed on `app.tenant_id`, so even
        # inserting the tenant's own row requires that GUC set to its own id.
        async with tenant_session(engine, DEMO_TENANT_ID) as session:
            stmt = (
                pg_insert(tenants)
                .values(
                    id=DEMO_TENANT_ID,
                    name=DEMO_TENANT_NAME,
                    org_ref=DEMO_ORG_REF,
                    status="active",
                )
                .on_conflict_do_nothing(index_elements=[tenants.c.id])
            )
            await session.execute(stmt)
            await session.commit()
    finally:
        await engine.dispose()
    print(f"seeded demo tenant {DEMO_TENANT_ID} (org_ref={DEMO_ORG_REF!r})")


def main() -> None:
    asyncio.run(seed())


if __name__ == "__main__":
    main()
