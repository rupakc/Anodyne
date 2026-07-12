# Anodyne

**Anodyne** is a cloud-agnostic, multi-tenant platform for **synthetic data generation** and
**LLM-as-a-Judge benchmarking**. It generates multimodal datasets from descriptions, templates, or
sample data; injects controlled noise, drift, outliers, and bias; exports to multiple formats; and
evaluates the results with a mixture-of-experts LLM judge pipeline — with human-in-the-loop review
throughout.

> **Project status — early foundation.** The **Platform Foundation + LLM Abstraction** walking
> skeleton is implemented (multi-tenant identity, storage/secrets, observability, and an
> LLM-agnostic layer behind a FastAPI gateway). Generation, perturbation, export, evaluation,
> the web UI, and full deployment are on the roadmap (see below) and not yet built.

## Why Anodyne

- **Generate** tabular, text, image, audio, and video datasets from plain-language descriptions,
  starter templates, or an uploaded sample.
- **Perturb** — inject noise, feature drift, outliers/anomalies, and targeted bias/edge cases.
- **Export** to CSV, JSON, Parquet, and Arrow (large datasets default to Parquet/Arrow).
- **Benchmark** with a 360° mixture-of-experts LLM-as-a-Judge evaluation and a report.
- **Bring your own model** — any cloud provider or local model (Ollama/vLLM), per tenant.
- **Human-in-the-loop** review, annotation, and feedback.
- **Multi-tenant, cloud-agnostic** — runs on GCP (Cloud Run / GKE / Cloud Functions) or on-prem.

## Architecture

Full detail in [`docs/architecture.md`](docs/architecture.md). In brief:

- **Backend:** Python 3.12 + FastAPI (async), Pydantic v2, hexagonal (ports & adapters).
- **Distributed compute:** Ray. **Orchestration:** Temporal (durable, HITL-pausable workflows).
- **Identity:** Keycloak (OIDC) + Postgres row-level security for tenant isolation.
- **Storage:** Postgres (metadata) · S3-compatible object store (MinIO on-prem / GCS cloud) · Redis.
- **LLM layer:** LiteLLM behind a thin `LLMProvider` port (100+ providers + local).
- **Frontend:** Next.js + TypeScript (autumn-pastel design system) — live (generation UI).

### Monorepo layout

```
apps/
  api-gateway/          FastAPI: OIDC auth, tenant resolution, RBAC, routes
packages/
  anodyne-core          domain models + ports (no infra imports)
  anodyne-tenancy       OIDC validation, role-based authorization
  anodyne-storage       Fernet secrets, tenant-prefixed S3, Postgres RLS sessions + Alembic
  anodyne-observability structlog JSON logging + OpenTelemetry
  anodyne-llm           LiteLLM adapter + DB-backed per-tenant model registry
infra/docker/           docker-compose backbone + Keycloak realm seed
docs/                   architecture, specs, plans, dev runbook
```

## Quick start (local dev)

Requires [uv](https://docs.astral.sh/uv/) and Docker.

```bash
uv sync                 # install the workspace
make up                 # start Postgres, Redis, MinIO, Keycloak
make migrate            # apply DB schema + RLS policies (runs as the postgres owner role)
make seed               # insert the demo tenant
uv run uvicorn api_gateway.app:create_app --factory --reload   # run the gateway
```

See [`docs/dev-runbook.md`](docs/dev-runbook.md) for obtaining a token from Keycloak and calling
`/me`, `/models`, and `/llm/invoke` end-to-end.

## Testing

```bash
uv run pytest -m "not integration"   # fast suite (no external services)
uv run pytest -m integration         # RLS + registry tests (requires Docker; run in CI)
uv run ruff check . && uv run mypy . # lint + strict type-check
```

CI (`.github/workflows/ci.yml`) runs the quality gate on every PR plus an integration job that
exercises Postgres row-level security via testcontainers.

## Multi-tenancy & security

- Every tenant-scoped table carries `tenant_id` with a Postgres RLS policy; the application connects
  as a **non-superuser** role so RLS is enforced, with explicit tenant filters as defense-in-depth.
- Bring-your-own model API keys are encrypted at rest (envelope encryption) — only encrypted refs
  are stored, never plaintext, and they are never returned by the API.

## Roadmap

| Stage | Sub-system | Status |
|------|------------|--------|
| A + B | Platform Foundation + LLM Abstraction | ✅ walking skeleton |
| C0 | Generation foundation — tabular-from-description, Temporal + Ray, Web UI | ✅ done |
| C1–C6 | Generation: tabular (full), text, image, audio, video, templates | in progress |
| D | Perturbation (noise, drift, outliers, bias/edge-case) | planned |
| E | Export & Storage (CSV/JSON/Parquet/Arrow) | planned |
| F | Evaluation Engine (LLM-as-a-Judge MoE + reports) | planned |
| G | Human-in-the-loop & Annotation | planned |
| H | Web UI | planned |
| I | Deployment & CI/CD (GCP + on-prem) | in progress |

## Documentation

- **Architecture & specs:** [`docs/architecture.md`](docs/architecture.md), `docs/superpowers/specs/`, `docs/superpowers/plans/`.
- **Local dev runbook:** [`docs/dev-runbook.md`](docs/dev-runbook.md).
- **Non-technical feature guides (GitHub Wiki):** source in [`docs/wiki/`](docs/wiki/) — plain-language explanations of each feature (Foundation, Bring Your Own AI Model, Multi-Tenancy & Security, Generation Engine, Local Development). Published to the repository Wiki.

## License

See [LICENSE](LICENSE).
