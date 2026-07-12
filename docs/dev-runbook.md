# Dev runbook: local backbone → `/llm/invoke`

Requires Docker (Desktop or Engine) with the `compose` plugin, plus `pnpm`
and `uv` on the host for `make dev`. This exact flow has **not** been
executed in the environment that authored these files (no Docker daemon
available there — see Task 12 report); validate it here before relying on
it.

## 1. Start the backbone

```bash
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# paste the output into ANODYNE_SECRET_KEY in .env

make up
```

`make up` now brings up the full backbone: Postgres, Redis (host port 6379),
MinIO, Keycloak, **Temporal** (+ Temporal UI on 8088), **Ray head** (dashboard
8265, client 10001), and **Ollama**.

> **Port note:** `ray-head`'s internal Redis (6379) is **not** published to the
> host — the generation-worker connects to Ray via the client port **10001**,
> so the app `redis` service keeps host port **6379** with no clash. The
> `config.py` defaults work out of the box for `make dev` once `make up` is
> running: `redis://localhost:6379/0`, `temporal_address=localhost:7233`, and
> `ray_address=ray://localhost:10001` (the `ray-head` container). Set
> `ANODYNE_RAY_ADDRESS=""` to run an embedded local Ray instead of the container.

Wait for Keycloak to finish importing the `anodyne` realm — tail its logs
until you see `Imported realm anodyne` (or just poll the realm's OIDC
discovery document):

```bash
until curl -sf http://localhost:8080/realms/anodyne/.well-known/openid-configuration \
    >/dev/null; do sleep 2; done
```

Also wait for Temporal to be ready before running the worker (`make dev`
starts `generation_worker.main`, which connects to Temporal on startup):

```bash
until curl -sf http://localhost:8088 >/dev/null; do sleep 2; done   # Temporal UI up
```

Ray's dashboard (http://localhost:8265) and the Ray client port (10001,
`ANODYNE_RAY_ADDRESS=ray://localhost:10001`) come up quickly once the
`ray-head` container is healthy; `docker compose ps` should show it as
`Up`.

## 2. Migrate + seed

```bash
make migrate   # alembic upgrade head against Postgres
make seed      # upserts the demo tenant (id 11111111-1111-1111-1111-111111111111)
```

Two Postgres roles are involved here, deliberately (see
`infra/docker/postgres/init/01-app-role.sql` and `.env.example`):

- `postgres` — the bootstrap SUPERUSER (owns the tables). `make migrate`
  connects as this role (`ANODYNE_DB_DSN`) because Alembic needs owner
  privileges to create/alter schema and RLS policies.
- `anodyne_app` — a non-superuser role, created by the init script the first
  time the `postgres` container starts against an empty volume. `make seed`
  and the api-gateway app itself connect as this role (`ANODYNE_DATABASE_URL`)
  so that Postgres row-level security is actually enforced at runtime:
  superusers (and any role with `BYPASSRLS`) ignore RLS even when a table has
  `FORCE ROW LEVEL SECURITY` set, so the app must never run as `postgres`.

The init script only runs against a fresh, empty data volume. If you're
reusing an existing `anodyne-postgres-data` volume from before this role
split, run `make down` (which removes volumes) and `make up` again so the
script executes.

## 3. Run the app processes

The api-gateway, generation-worker, and web app are not in
`docker-compose.yml` (only the backbone is) — they run on the host against
the backbone. Run all three concurrently with:

```bash
make dev
```

This runs, in parallel (Ctrl-C stops all three):

- `uv run uvicorn api_gateway.app:create_app --factory --reload --port 8000`
- `uv run python -m generation_worker.main` (the Temporal worker for
  `GenerationWorkflow`; requires Temporal + Ray + Postgres/Redis/MinIO to
  already be up)
- `pnpm --dir apps/web dev` (the web UI)

Before running the web app, create `apps/web/.env.local` (git-ignored, not
checked in) with the Auth.js / Keycloak login config it reads from
`process.env` (see `apps/web/auth.ts`):

```bash
# apps/web/.env.local
AUTH_SECRET=<openssl rand -base64 32>
KEYCLOAK_ISSUER=http://localhost:8080/realms/anodyne     # default if unset
KEYCLOAK_CLIENT_ID=anodyne                                # default if unset
KEYCLOAK_CLIENT_SECRET=dev-only-anodyne-client-secret     # from infra/docker/keycloak/anodyne-realm.json, dev-only
```

`AUTH_SECRET` and `KEYCLOAK_CLIENT_SECRET` are required — without them,
Auth.js will fail to sign session JWTs / authenticate against Keycloak.
`KEYCLOAK_ISSUER` and `KEYCLOAK_CLIENT_ID` fall back to the defaults shown
above (the local `anodyne` realm/client), so they only need to be set to
point at a different Keycloak instance.

If you only need the gateway (e.g. for the `curl` walkthrough below),
running the first command by itself is enough.

## 4. Get a token for the demo user

The `anodyne` client is confidential with direct-access-grants enabled, so a
password grant works for local testing:

```bash
TOKEN=$(curl -s -X POST \
  http://localhost:8080/realms/anodyne/protocol/openid-connect/token \
  -H 'Content-Type: application/x-www-form-urlencoded' \
  -d grant_type=password \
  -d client_id=anodyne \
  -d client_secret=dev-only-anodyne-client-secret \
  -d username=demo@anodyne.dev \
  -d password=demo \
  -d scope=openid \
  | jq -r .access_token)
```

The resulting access token carries `aud: ["anodyne"]` (audience mapper) and
`org_id: "11111111-1111-1111-1111-111111111111"` (user-attribute mapper),
which is exactly the tenant id `make seed` inserts.

## 5. `GET /me` — expect the demo tenant/roles

```bash
curl -s http://localhost:8000/me -H "Authorization: Bearer $TOKEN" | jq .
```

Expect `tenant_id: "11111111-1111-1111-1111-111111111111"` and
`roles: ["admin"]`.

## 6. Register a model and invoke it

```bash
CONFIG_ID=$(curl -s -X POST http://localhost:8000/models \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"gpt4o","provider":"openai","model":"gpt-4o","api_key":"sk-REPLACE-ME"}' \
  | jq -r .id)

curl -s http://localhost:8000/models -H "Authorization: Bearer $TOKEN" | jq .

curl -s -X POST http://localhost:8000/llm/invoke \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d "{\"model_config_id\":\"$CONFIG_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}]}" \
  | jq .
```

`/llm/invoke` proxies through `LiteLLMProvider` (`packages/anodyne-llm`), so
a real response requires a valid upstream provider API key in place of
`sk-REPLACE-ME`; with a fake key the call reaches LiteLLM and fails with a
provider auth error rather than a 503/404 — that failure mode is itself
sufficient to confirm the gateway → registry → provider wiring is live.

### External-model path (bring your own API key)

Use this when you have a real upstream provider key (OpenAI, Anthropic,
etc.) — this is what step 6 above does. `provider` + `model` are passed
straight through to LiteLLM (`f"{provider}/{model}"`), and `api_key` is
stored via `secret_ref` (never returned by `/models`). Any LiteLLM-supported
provider works, e.g. `{"provider":"anthropic","model":"claude-3-5-sonnet-20241022","api_key":"sk-ant-..."}`.

### Offline path (local Ollama, no external API key)

For fully offline/local generation, use the `ollama` container started by
`make up` (http://localhost:11434, `ANODYNE_OLLAMA_BASE` in
`.env.example`):

```bash
# pull a model into the ollama container's volume (one-time, ~4-5GB for llama3)
docker compose -f infra/docker/docker-compose.yml exec ollama ollama pull llama3

# register it as a model config for the demo tenant — no api_key needed,
# api_base points at the Ollama container; LiteLLM routes "ollama/llama3"
# to it.
curl -s -X POST http://localhost:8000/models \
  -H "Authorization: Bearer $TOKEN" -H 'Content-Type: application/json' \
  -d '{"name":"llama3-local","provider":"ollama","model":"llama3","api_base":"http://localhost:11434"}'
```

Then swap `model_config_id` in the `/llm/invoke` call from step 6 to this
config's `id` to get a real (local, no-cost) completion — no upstream API
key required, which is the point: the demo tenant is usable end-to-end
without any external provider credentials.

## 7. Tear down

```bash
make down   # stops containers and removes volumes
```
