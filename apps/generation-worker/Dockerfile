# syntax=docker/dockerfile:1
#
# generation-worker — Temporal worker dispatching Ray generation activities
# (see apps/generation-worker/src/generation_worker/main.py, entrypoint
# `python -m generation_worker.main`). Long-running process, no HTTP surface.
#
# Build from the REPO ROOT so the uv workspace resolves:
#   docker build -f apps/generation-worker/Dockerfile -t anodyne/generation-worker .
FROM python:3.12-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.9.21 /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Build tooling for transitive scientific-stack deps (Ray, CTGAN/TVAE, SDXL/
# TTS/video model plumbing) without a prebuilt manylinux cp312 wheel.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# See apps/api-gateway/Dockerfile for why the full packages/ tree + both
# Python apps are copied even though only generation-worker's own deps get
# installed below.
COPY pyproject.toml uv.lock uv.toml ./
COPY packages ./packages
COPY apps/api-gateway ./apps/api-gateway
COPY apps/generation-worker ./apps/generation-worker

RUN uv sync --frozen --no-dev --no-editable --package generation-worker

FROM python:3.12-slim AS runtime

# libgomp1: OpenMP runtime for numpy/scipy/Ray/CTGAN-TVAE at import time;
# ca-certificates: TLS to Postgres/S3/Temporal/Redis/LLM + media providers.
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

# No EXPOSE: this process is a Temporal/Ray client, not an HTTP server.
# Liveness for this container is process-alive (see infra/k8s + docs/deployment.md).
CMD ["python", "-m", "generation_worker.main"]
