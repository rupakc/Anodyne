# syntax=docker/dockerfile:1
#
# api-gateway — FastAPI OIDC gateway (see apps/api-gateway/src/api_gateway/app.py,
# launched via `uvicorn api_gateway.app:create_app --factory`).
#
# Build from the REPO ROOT so the uv workspace resolves:
#   docker build -f apps/api-gateway/Dockerfile -t anodyne/api-gateway .
#
# Two stages: `builder` resolves + installs the uv workspace into a venv
# (using `--package api-gateway` so only its transitive deps are pulled in,
# and `--no-editable` so workspace packages are installed as real wheels,
# not path references back to source); `runtime` copies just that venv onto
# a slim base and runs as a non-root user. No compilers, no source tree, no
# uv binary ship to production.
FROM python:3.12-slim AS builder

# Pin uv via its official distroless image (see
# https://docs.astral.sh/uv/guides/integration/docker/) rather than `pip
# install uv`, so the resolver itself is reproducible across CI/local.
COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Build tooling for any transitive scientific-stack dependency (tabular
# synthesis: CTGAN/TVAE/SDV) without a prebuilt manylinux cp312 wheel. Not
# present in the runtime stage below.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# `uv sync --package api-gateway` still needs every workspace member that
# api-gateway depends on (path deps declared in [tool.uv.sources] —
# pyproject.toml) resolvable on disk, so the full `packages/` tree and both
# Python apps come along; apps/web is excluded from the uv workspace itself
# (pyproject.toml `[tool.uv.workspace] exclude`) and isn't needed here.
COPY pyproject.toml uv.lock uv.toml ./
COPY packages ./packages
COPY apps/api-gateway ./apps/api-gateway
COPY apps/generation-worker ./apps/generation-worker

RUN uv sync --frozen --no-dev --no-editable --package api-gateway

FROM python:3.12-slim AS runtime

# libgomp1: OpenMP runtime needed by numpy/scipy/CTGAN-TVAE wheels at import
# time; ca-certificates: TLS to Postgres/S3/Keycloak/LLM providers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system anodyne \
    && useradd --system --gid anodyne --no-create-home --shell /usr/sbin/nologin anodyne

ENV PATH="/app/.venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY --from=builder --chown=anodyne:anodyne /app/.venv /app/.venv

USER anodyne
EXPOSE 8080

# Cloud Run injects $PORT (defaults to 8080, matching the Kubernetes/Compose
# manifests under infra/); `sh -c` so ${PORT:-8080} expands at container
# start rather than at image build time.
CMD ["sh", "-c", "exec uvicorn api_gateway.app:create_app --factory --host 0.0.0.0 --port ${PORT:-8080}"]
