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
#
# Two separate DSNs, deliberately: `migrate` runs Alembic against
# ANODYNE_DB_DSN (the `postgres` superuser — owns the tables/RLS policies);
# `seed` and the app itself run against ANODYNE_DATABASE_URL (the
# non-superuser `anodyne_app` role — see infra/docker/postgres/init and
# .env.example). Superusers bypass row-level security even with FORCE ROW
# LEVEL SECURITY, so the app must never use the migration DSN.
-include .env
export

.PHONY: up down migrate seed test dev

up:        ; docker compose -f infra/docker/docker-compose.yml up -d
down:      ; docker compose -f infra/docker/docker-compose.yml down -v
migrate:   ; uv run alembic -c packages/anodyne-storage/alembic.ini upgrade head
seed:      ; uv run python -m api_gateway.seed
test:      ; uv run pytest

# Runs the three host-side dev processes concurrently: the api-gateway
# (uvicorn, reload on), the generation-worker (Temporal worker process), and
# the web app (pnpm dev server). Assumes `make up` + `make migrate` + `make
# seed` have already been run so Postgres/Redis/MinIO/Keycloak/Temporal/
# Ray/Ollama are up. Ctrl-C stops all three (the `trap`/`wait` combo below
# forwards SIGINT to the child processes and waits for them to exit).
#
# NOTE: `apps/web` doesn't exist yet as of this commit (it lands in a later
# task); until then the third leg will fail with a "no such directory"
# error while the gateway and worker legs still start fine.
#
# If you'd rather run these in separate terminals instead, just run each
# command on the right of `;` by hand.
dev:
	@trap 'kill 0' EXIT INT TERM; \
	uv run uvicorn api_gateway.app:create_app --factory --reload --port 8000 & \
	uv run python -m generation_worker.main & \
	pnpm --dir apps/web dev & \
	wait
