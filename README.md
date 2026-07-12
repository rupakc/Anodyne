# Anodyne

**Anodyne** is a cloud-agnostic, multi-tenant platform for **synthetic data generation** and
**LLM-as-a-Judge benchmarking**. It generates multimodal datasets from descriptions, templates, or
sample data; injects controlled noise, drift, outliers, and bias; exports to multiple formats; and
evaluates the results with a mixture-of-experts LLM judge pipeline — with human-in-the-loop review
throughout.

> **Project status.** The **Platform Foundation + LLM Abstraction** and the **Generation Engine**
> (all five modalities) are implemented: multi-tenant identity, storage/secrets, observability, an
> LLM-agnostic layer, and dataset generation for **tabular (from description *and* sample), text,
> image, audio, and video** — orchestrated by Temporal + Ray, with starter templates and
> bias/edge-case steering, driven from a Next.js web UI. Deployment & CI/CD — Dockerfiles, a
> build/SBOM/scan/push pipeline, and Cloud Run/GKE/on-prem manifests — is scaffolded (see
> `docs/deployment.md`); wiring it up against a real GCP project is a follow-up. Perturbation,
> export, evaluation, and human-in-the-loop are still on the roadmap (see below).

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
  api-gateway/          FastAPI: OIDC auth, tenant resolution, RBAC, dataset + provider routes
  generation-worker/    Temporal worker dispatching Ray generation activities
  web/                  Next.js UI (Auth.js/Keycloak, autumn-pastel) — create → generate → download
packages/
  anodyne-core          shared base models + ports
  anodyne-tenancy       OIDC validation, role-based authorization
  anodyne-storage       Fernet secrets, tenant-prefixed S3, Postgres RLS sessions + Alembic
  anodyne-observability structlog JSON logging + OpenTelemetry
  anodyne-llm           LiteLLM adapter + DB-backed per-tenant model registry
  anodyne-dataset       dataset/profile/job/version domain models + ports (Generator, ...)
  anodyne-generation    LLM schema proposer + deterministic tabular sampler
  anodyne-tabular       from-sample profiling + copula/CTGAN/TVAE (+ SDV opt-in)
  anodyne-text          LLM text corpora (classification/QA/summarization/chat)
  anodyne-image         provider-agnostic image generation (SDXL / external APIs)
  anodyne-audio         provider-agnostic audio/TTS generation
  anodyne-video         provider-agnostic text-to-video generation
  anodyne-templates     starter templates + bias/edge-case/use-case directives
  anodyne-perturbation  noise/drift/outlier/bias/edge-case perturbation (tabular + text)
  anodyne-export        CSV/JSON/Parquet/Arrow chunked export + presigned download
  anodyne-evaluation    LLM-as-a-Judge mixture-of-experts 360° evaluation + report
  anodyne-compute       Ray shard-generation tasks + GPU actor seams
  anodyne-workflows     Temporal GenerationWorkflow + modality-registry activities
infra/docker/           docker-compose backbone (Postgres, Redis, MinIO, Keycloak, Temporal, Ray, Ollama)
docs/                   architecture, specs, plans, dev runbook, wiki
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
| C1 | Tabular (full) — from-sample profiling, copula/CTGAN/TVAE, SDV opt-in | ✅ done |
| C2 | Text — LLM corpora (classification/QA/summarization/chat), dedup/quality | ✅ done |
| C3 | Image — provider-agnostic (SDXL / external APIs) | ✅ done |
| C4 | Audio — provider-agnostic TTS (self-hosted / external) | ✅ done |
| C5 | Video — provider-agnostic text-to-video | ✅ done |
| C6 | Starter templates + bias/edge-case/use-case directives | ✅ done |
| D | Perturbation — noise, drift, outliers/anomalies, bias/edge-case (tabular + text, deterministic) | ✅ done |
| E | Export & Storage — CSV/JSON/Parquet/Arrow, chunked/streamed, >500K→Parquet default | ✅ done |
| F | Evaluation Engine — LLM-as-a-Judge MoE (fidelity/diversity/privacy/utility/bias/qualitative) + JSON+HTML report | ✅ done |
| G | Human-in-the-loop & Annotation | planned |
| H | Web UI (full, across all modalities + D/E/F) | planned |
| I | Deployment & CI/CD (GCP + on-prem) | ✅ Dockerfiles, CI/CD (build+SBOM+Trivy+WIF push), Cloud Run/GKE manifests, Terraform skeleton — see [`docs/deployment.md`](docs/deployment.md) |

## Documentation

- **Architecture & specs:** [`docs/architecture.md`](docs/architecture.md), `docs/superpowers/specs/`, `docs/superpowers/plans/`.
- **Local dev runbook:** [`docs/dev-runbook.md`](docs/dev-runbook.md).
- **Deployment & CI/CD:** [`docs/deployment.md`](docs/deployment.md) — local → on-prem → Cloud Run → GKE, and the secret-management model.
- **Non-technical feature guides (GitHub Wiki):** source in [`docs/wiki/`](docs/wiki/) — plain-language explanations of each feature (Foundation, Bring Your Own AI Model, Multi-Tenancy & Security, Generation Engine, Local Development, Deployment). Published to the repository Wiki.

## License

See [LICENSE](LICENSE).
