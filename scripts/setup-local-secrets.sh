#!/usr/bin/env bash
#
# setup-local-secrets.sh
#
# Populate the local dev env files with every secret/config value the Anodyne
# stack needs, so the gateway, workers, and web survive restarts without the
# "MissingSecret / NoCredentialsError / Failed to fetch" class of errors.
#
# Writes:
#   .env                 -> backend (api-gateway, generation-worker, evaluation-worker)
#   apps/web/.env.local  -> frontend (Next.js / NextAuth -> Keycloak)
#
# Sources (in order of authority):
#   * the running dev stack's process environment  -> the EXACT working Fernet
#     key, MinIO/S3 creds, and LLM API keys (a fresh Fernet key would break
#     decryption of already-stored provider secrets, so we must reuse the live one)
#   * infra/docker/keycloak/anodyne-realm.json      -> the dev Keycloak client secret
#   * openssl                                        -> a fresh AUTH_SECRET (only if absent)
#
# Properties: idempotent (re-runnable), backs up existing files, chmod 600,
# and NEVER prints a secret value to the terminal.
#
# Run it WHILE your current dev stack is up (the Fernet key + LLM keys live only
# in those processes). Usage:
#     bash scripts/setup-local-secrets.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

ENV_FILE="$ROOT/.env"
WEB_ENV="$ROOT/apps/web/.env.local"
REALM="$ROOT/infra/docker/keycloak/anodyne-realm.json"

log()  { printf '   %s\n' "$*"; }
head() { printf '\n==> %s\n' "$*"; }

# ---- upsert KEY VALUE FILE : replace-or-append a single-quoted line ---------
# (single-quote so base64 '+/=' survive `set -a; . .env`; strips any prior line
#  for KEY first so re-runs never duplicate)
upsert() {
  local key="$1" val="$2" file="$3"
  touch "$file"
  grep -v -E "^${key}=" "$file" > "$file.tmp" 2>/dev/null || true
  printf "%s='%s'\n" "$key" "$val" >> "$file.tmp"
  mv "$file.tmp" "$file"
}

backup() {
  if [ -f "$1" ]; then
    cp "$1" "$1.bak.$(date +%Y%m%d-%H%M%S)"
    log "backed up existing $1"
  fi
}

# ---- read one env var out of a running process (empty if absent) -----------
getenv() { ps eww "$1" 2>/dev/null | tr ' ' '\n' | grep -E "^$2=" | head -1 | cut -d= -f2- || true; }

# ---- find a live stack process that carries the backend secrets ------------
find_src_pid() {
  local pat pid
  for pat in "generation_worker.main" "evaluation_worker.main" "uvicorn api_gateway"; do
    for pid in $(pgrep -f "$pat" 2>/dev/null || true); do
      if ps eww "$pid" 2>/dev/null | tr ' ' '\n' | grep -q '^ANODYNE_SECRET_KEY='; then
        echo "$pid"; return 0
      fi
    done
  done
  return 1
}

head "Recovering backend secrets from the running stack"
SRC_PID="$(find_src_pid || true)"
if [ -z "${SRC_PID:-}" ]; then
  cat >&2 <<'ERR'
ERROR: could not find a running Anodyne process exposing ANODYNE_SECRET_KEY.

The existing Fernet key and LLM API keys live ONLY in the environment of your
running gateway/workers. Start your current dev stack first (so at least one of
api-gateway / generation-worker / evaluation-worker is running with those vars),
then re-run this script. Do NOT generate a new Fernet key — it cannot decrypt
provider secrets already stored in the DB.
ERR
  exit 1
fi
log "source process: pid $SRC_PID"

SECRET_KEY="$(getenv "$SRC_PID" ANODYNE_SECRET_KEY)"
GEMINI="$(getenv "$SRC_PID" ANODYNE_GEMINI_API_KEY)"
ANTHROPIC="$(getenv "$SRC_PID" ANODYNE_ANTHROPIC_API_KEY)"
RAY_ADDR="$(getenv "$SRC_PID" ANODYNE_RAY_ADDRESS)"
AWS_AK="$(getenv "$SRC_PID" AWS_ACCESS_KEY_ID)"
AWS_SK="$(getenv "$SRC_PID" AWS_SECRET_ACCESS_KEY)"
AWS_EP="$(getenv "$SRC_PID" AWS_ENDPOINT_URL)"
AWS_EPS3="$(getenv "$SRC_PID" AWS_ENDPOINT_URL_S3)"
AWS_REGION="$(getenv "$SRC_PID" AWS_DEFAULT_REGION)"

[ -n "$SECRET_KEY" ] || { echo "ERROR: ANODYNE_SECRET_KEY came back empty." >&2; exit 1; }

head "Reading the Keycloak dev client secret from the realm file"
[ -f "$REALM" ] || { echo "ERROR: realm file not found: $REALM" >&2; exit 1; }
KC_SECRET="$(python3 -c "import json,sys;print(next(c['secret'] for c in json.load(open('$REALM'))['clients'] if c['clientId']=='anodyne'))")"
[ -n "$KC_SECRET" ] || { echo "ERROR: no 'anodyne' client secret in realm file." >&2; exit 1; }

head "Resolving AUTH_SECRET (NextAuth session key)"
existing_auth() { [ -f "$WEB_ENV" ] && grep -E "^AUTH_SECRET=" "$WEB_ENV" | head -1 | cut -d= -f2- | tr -d "'\""; }
AUTH_SECRET="$(existing_auth || true)"
if [ -z "${AUTH_SECRET:-}" ]; then
  AUTH_SECRET="$(openssl rand -base64 32 2>/dev/null || python3 -c 'import secrets,base64;print(base64.b64encode(secrets.token_bytes(32)).decode())')"
  log "generated a new AUTH_SECRET (existing web sessions will need one re-login)"
else
  log "kept the existing AUTH_SECRET (sessions stay valid)"
fi

head "Writing $ENV_FILE  (backend)"
backup "$ENV_FILE"
upsert ANODYNE_SECRET_KEY "$SECRET_KEY" "$ENV_FILE";        log "ANODYNE_SECRET_KEY"
[ -n "$GEMINI" ]     && { upsert ANODYNE_GEMINI_API_KEY    "$GEMINI"     "$ENV_FILE"; log "ANODYNE_GEMINI_API_KEY"; }
[ -n "$ANTHROPIC" ]  && { upsert ANODYNE_ANTHROPIC_API_KEY "$ANTHROPIC"  "$ENV_FILE"; log "ANODYNE_ANTHROPIC_API_KEY"; }
# Embedded Ray for local dev (no ray-head container in `make up`); a real
# ray://host:port address is only used in a distributed deployment. Kept only if
# the captured value is a valid ray:// URL, else blanked to force embedded.
case "$RAY_ADDR" in ray://*) : ;; *) RAY_ADDR="" ;; esac
upsert ANODYNE_RAY_ADDRESS "$RAY_ADDR" "$ENV_FILE"; log "ANODYNE_RAY_ADDRESS (embedded)"
[ -n "$AWS_AK" ]     && { upsert AWS_ACCESS_KEY_ID         "$AWS_AK"     "$ENV_FILE"; log "AWS_ACCESS_KEY_ID"; }
[ -n "$AWS_SK" ]     && { upsert AWS_SECRET_ACCESS_KEY     "$AWS_SK"     "$ENV_FILE"; log "AWS_SECRET_ACCESS_KEY"; }
[ -n "$AWS_EP" ]     && { upsert AWS_ENDPOINT_URL          "$AWS_EP"     "$ENV_FILE"; log "AWS_ENDPOINT_URL"; }
[ -n "$AWS_EPS3" ]   && { upsert AWS_ENDPOINT_URL_S3       "$AWS_EPS3"   "$ENV_FILE"; log "AWS_ENDPOINT_URL_S3"; }
[ -n "$AWS_REGION" ] && { upsert AWS_DEFAULT_REGION        "$AWS_REGION" "$ENV_FILE"; log "AWS_DEFAULT_REGION"; }
chmod 600 "$ENV_FILE"

head "Writing $WEB_ENV  (frontend)"
mkdir -p "$(dirname "$WEB_ENV")"
backup "$WEB_ENV"
upsert AUTH_SECRET            "$AUTH_SECRET"                              "$WEB_ENV"; log "AUTH_SECRET"
upsert KEYCLOAK_CLIENT_SECRET "$KC_SECRET"                               "$WEB_ENV"; log "KEYCLOAK_CLIENT_SECRET"
upsert KEYCLOAK_ISSUER        "http://localhost:8080/realms/anodyne"     "$WEB_ENV"; log "KEYCLOAK_ISSUER"
upsert KEYCLOAK_CLIENT_ID     "anodyne"                                  "$WEB_ENV"; log "KEYCLOAK_CLIENT_ID"
# Only needed if the gateway does NOT run on :8000 (the web default). Uncomment
# and adjust if you run the gateway on a different port:
# upsert NEXT_PUBLIC_API_BASE "http://localhost:8001" "$WEB_ENV"
chmod 600 "$WEB_ENV"

head "Verification (key present? — values never shown)"
for k in ANODYNE_SECRET_KEY AWS_ACCESS_KEY_ID AWS_SECRET_ACCESS_KEY AWS_ENDPOINT_URL AWS_DEFAULT_REGION ANODYNE_GEMINI_API_KEY ANODYNE_RAY_ADDRESS; do
  grep -qE "^$k=" "$ENV_FILE" && log ".env               $k  ✓" || log ".env               $k  (absent — not set on source process)"
done
for k in AUTH_SECRET KEYCLOAK_CLIENT_SECRET KEYCLOAK_ISSUER KEYCLOAK_CLIENT_ID; do
  grep -qE "^$k=" "$WEB_ENV" && log "apps/web/.env.local $k  ✓" || log "apps/web/.env.local $k  (absent)"
done

cat <<'NOTE'

Done. Both files are chmod 600 and git-ignored (nothing gets committed).

HOW THESE TAKE EFFECT
  Frontend: Next.js auto-loads apps/web/.env.local — just restart the web dev server.
  Backend:  the gateway/workers read ANODYNE_* from .env via pydantic, but the
            AWS_* (MinIO) creds are read by boto3 from the process ENVIRONMENT,
            not from the .env file. So the backend must be launched with .env
            EXPORTED into the environment. The easiest way is:

                bash scripts/dev-run.sh

            which exports .env and launches the gateway (:8000), both workers,
            and web. Or do it by hand:

                set -a; . ./.env; set +a
                uv run uvicorn api_gateway.app:create_app --factory --host 0.0.0.0 --port 8000 &
                uv run python -m generation_worker.main &
                uv run python -m evaluation_worker.main &
                npm --prefix apps/web run dev &
NOTE
