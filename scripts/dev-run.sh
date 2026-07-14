#!/usr/bin/env bash
#
# dev-run.sh
#
# Launch the full local Anodyne app tier with the backend secrets exported into
# the environment (so boto3 sees the MinIO/S3 creds — the .env file alone is not
# enough for boto3). Assumes the Docker backbone (`make up`) is already running
# and that scripts/setup-local-secrets.sh has populated .env + apps/web/.env.local.
#
#   Ports: gateway :8000 (matches the web's NEXT_PUBLIC_API_BASE default),
#          web :3000. Ctrl-C stops everything.
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [ ! -f ./.env ]; then
  echo "No .env found. Run:  bash scripts/setup-local-secrets.sh   (while your current stack is up)" >&2
  exit 1
fi

# Export every var from .env into the environment (boto3 + pydantic + LiteLLM all read it).
set -a
# shellcheck disable=SC1091
. ./.env
set +a

# Ray address: this stack uses EMBEDDED Ray (no ray-head container in `make up`).
# Keep ANODYNE_RAY_ADDRESS only if it's a real Ray client address (ray://...);
# otherwise blank it so ray_init() starts a local embedded instance. This also
# neutralizes a malformed value that setup-local-secrets.sh may have captured.
case "${ANODYNE_RAY_ADDRESS:-}" in
  ray://*) : ;;
  *) export ANODYNE_RAY_ADDRESS="" ;;
esac

echo "Launching gateway (:8000), generation-worker, evaluation-worker, web (:3000)..."
echo "(Ctrl-C stops all)"

# Kill the whole process group on exit so no orphans linger.
trap 'kill 0' EXIT INT TERM

uv run uvicorn api_gateway.app:create_app --factory --host 0.0.0.0 --port 8000 &
uv run python -m generation_worker.main &
uv run python -m evaluation_worker.main &
npm --prefix apps/web run dev &

wait
