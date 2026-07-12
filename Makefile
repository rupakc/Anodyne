# Anodyne local dev backbone.
#
# `make up` boots Postgres/Redis/MinIO/Keycloak (Keycloak self-seeds the
# `anodyne` realm from infra/docker/keycloak/anodyne-realm.json via
# --import-realm). `make migrate` then applies Alembic migrations and
# `make seed` upserts the demo tenant so the demo Keycloak user's `org_id`
# claim resolves to a real row. See docs/dev-runbook.md for the full flow.
#
# `-include .env` + `export` make any vars copied from .env.example (DB DSN,
# OIDC settings, ANODYNE_SECRET_KEY, ...) visible to the recipes below,
# mirroring how the api-gateway process picks them up at runtime.
-include .env
export

.PHONY: up down migrate seed test

up:        ; docker compose -f infra/docker/docker-compose.yml up -d
down:      ; docker compose -f infra/docker/docker-compose.yml down -v
migrate:   ; uv run alembic -c packages/anodyne-storage/alembic.ini upgrade head
seed:      ; uv run python -m api_gateway.seed
test:      ; uv run pytest
