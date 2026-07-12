# Anodyne — Developer Guide

Synthetic data generation + LLM-as-a-Judge benchmarking platform. See `docs/architecture.md`.

## Commands
- Install: `uv sync`
- Test: `uv run pytest`
- Lint/format: `uv run ruff check --fix . && uv run ruff format .`
- Types: `uv run mypy .`
- Local backbone: `make up` (Postgres, Redis, Keycloak, MinIO)

## Conventions
- Hexagonal: domain + ports in `anodyne-core`; adapters in sibling packages; wiring in `apps/`.
- No adapter imports in `anodyne-core`.
- TDD: failing test first. Conventional commits. `mypy --strict` and `ruff` must pass.
- Multi-tenant: every tenant row carries `tenant_id` + an RLS policy; never log/store plaintext secrets.
