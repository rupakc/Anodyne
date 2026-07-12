# Dev runbook: local backbone → `/llm/invoke`

Requires Docker (Desktop or Engine) with the `compose` plugin. This exact
flow has **not** been executed in the environment that authored these files
(no Docker daemon available there — see Task 12 report); validate it here
before relying on it.

## 1. Start the backbone

```bash
cp .env.example .env
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
# paste the output into ANODYNE_SECRET_KEY in .env

make up
```

Wait for Keycloak to finish importing the `anodyne` realm — tail its logs
until you see `Imported realm anodyne` (or just poll the realm's OIDC
discovery document):

```bash
until curl -sf http://localhost:8080/realms/anodyne/.well-known/openid-configuration \
    >/dev/null; do sleep 2; done
```

## 2. Migrate + seed

```bash
make migrate   # alembic upgrade head against Postgres
make seed      # upserts the demo tenant (id 11111111-1111-1111-1111-111111111111)
```

## 3. Run the API gateway

The gateway app itself is not in `docker-compose.yml` (only the backbone
is); run it on the host against the backbone:

```bash
uv run uvicorn api_gateway.app:create_app --factory --port 8000
```

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

## 7. Tear down

```bash
make down   # stops containers and removes volumes
```
