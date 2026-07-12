# Anodyne — Deployment & CI/CD (Sub-system I) Design

- **Date:** 2026-07-12
- **Status:** Accepted
- **Roadmap:** Sub-system I (final roadmap row — infra only, no application logic)
- **Depends on:** `docs/architecture.md` §3 (service topology), §5 (IaC/CI-CD), §7 (roadmap);
  the existing `.github/workflows/ci.yml` quality gate; Alembic head `0006`
  (unchanged — this sub-system adds no migration).

## Goal

Take the three deployable units (api-gateway, generation-worker, web) from "runs on a laptop via
`make dev`" to "buildable as a container, scanned, pushed, and deployable to Cloud Run, GKE, or
on-prem Docker Compose" — without touching any application source. Everything here is
Dockerfiles, CI/CD YAML, deployment manifests, Terraform, and docs.

## Constraints this design honors

- No edits under `packages/**/src` or `apps/{api-gateway,generation-worker}/src` (the constraint's
  `apps/*/src` pattern doesn't cover `apps/web`, which has no `src/` dir at all — see below).
- One deliberate, minimal exception: `apps/web/next.config.ts` gains `output: "standalone"`. This
  is Next.js *build-output* configuration (which files `next build` emits), not application
  behavior — `next dev`/`next start` are unaffected. Standalone output is what makes a
  minimal, `node_modules`-free production image possible; there is no way to build the image the
  task asked for without it. No other web source file changed.
- No Alembic migration added; DB schema stays at head `0006`.
- No secret *values* anywhere — every credential is a Secret Manager / K8s Secret / `.env`
  *reference*, and `.env*` files are untouched.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Dockerfile build context | Repo root for all three (`docker build -f apps/X/Dockerfile .`) | `uv sync --package X` still needs every path-dependency in `[tool.uv.sources]` resolvable on disk, even ones not ultimately installed; a narrower context breaks resolution. One root `.dockerignore` keeps this consistent and keeps `.env*` out of every build context as defense-in-depth. |
| Python image strategy | Multi-stage: `builder` (python:3.12-slim + pinned `uv` binary + build-essential) → `runtime` (python:3.12-slim, no compiler, non-root) | `uv sync --frozen --no-dev --no-editable --package <app>` builds workspace-local packages as real wheels installed into `/app/.venv`; runtime stage copies only that venv — no source tree, no uv, no compiler ships. |
| Web image strategy | Multi-stage Node 22: `deps` (pnpm install, frozen lockfile) → `builder` (`next build`, standalone output) → `runtime` (only `.next/standalone` + `.next/static` + `public`) | Standalone output is self-contained (`server.js` + pruned `node_modules`); runtime stage never installs pnpm or reruns `pnpm install`. |
| Non-root everywhere | Dedicated `anodyne` system user (Python images) / built-in `node` user (web image); K8s manifests additionally set `readOnlyRootFilesystem: true` + drop all capabilities, with an `emptyDir` at `/tmp` (`HOME=/tmp`) for third-party scratch use (Ray, HF/torch caches) | Matches CLAUDE.md's security posture; `readOnlyRootFilesystem` needed a writable `/tmp` because several transitive deps (Ray, torch-backed CTGAN/TVAE/SDXL/TTS) write scratch files even though no first-party Anodyne code does (verified: no `tempfile`/`/tmp` usage in `packages/*/src` or `apps/*/src`). |
| generation-worker on Cloud Run | Modeled as a Cloud Run **Service** (not a Job), `ingress: internal`, `minScale: 1`, `cpu-throttling: false`, `containerConcurrency: 1` | It's a long-lived Temporal task-queue consumer, not request-driven or run-to-completion — neither a public Cloud Run Service nor a Cloud Run Job fits exactly; an always-on internal Service is the closest primitive. Documented in `docs/deployment.md` as the lighter-weight option; GKE + KubeRay is the recommended target once a tenant needs real Ray-cluster/GPU scale for image/audio/video. |
| GKE Ray cluster | `infra/k8s/base/raycluster.yaml` (KubeRay CRD) present but **not** in any `kustomization.yaml` resource list | KubeRay's operator/CRDs are a cluster-scoped one-time Helm install; `kustomize build` must succeed against a cluster that doesn't have it yet. Applied separately once the operator exists. |
| Kustomize `images[].newName` placeholders | Left as literal `${VAR}` strings, piped through `envsubst` by CI before `kubectl apply` | Kustomize doesn't do shell interpolation; this mirrors the same pattern already needed for `infra/cloudrun/*.yaml`, keeping one substitution story across both deploy targets. `kustomize build` still succeeds and was verified (see Validation). |
| On-prem prod compose | `infra/docker/docker-compose.prod.yml` as an **overlay** (`-f docker-compose.yml -f docker-compose.prod.yml`), not a rewrite | Keeps the dev backbone as the single source of truth for service definitions; the prod file only patches (restart policies, hidden ports, resource limits) and adds the three app containers as built images. Keycloak's `start-dev` command is deliberately left untouched — hardening it for real production (external DB, `start`, TLS) is a Keycloak-operations concern, not part of Anodyne's own deployment, and changing a third-party service's runtime mode is a bigger step than "layering on the backbone." Flagged as a follow-up in `docs/deployment.md`. |
| CI/CD auth | GCP Workload Identity Federation (`google-github-actions/auth`, `workload_identity_provider` input) exchanging GitHub's OIDC token for short-lived credentials | Task requirement; no Service Account JSON key ever exists. `attribute_condition` on the WIF provider scopes trust to exactly one `owner/repo`; a separate `ref`-scoped IAM binding further restricts which branches/tags can impersonate the deploy SA (`infra/terraform/workload_identity.tf`). |
| SBOM + scan gate | `anchore/sbom-action` (syft, SPDX-JSON) then `aquasecurity/trivy-action`, `exit-code: 1` on HIGH/CRITICAL, run against the **locally built, unpushed** image (`load: true`) before the separate push step | Fails the run before anything reaches Artifact Registry; SBOM uploaded as a build artifact regardless of scan outcome (audit trail even for a failed/fixed-forward build). | 
| Existing `ci.yml` | Left unmodified | Already satisfies the "quality gate" requirement exactly as specified (uv sync/ruff/mypy/pytest + testcontainers integration job + pnpm lint/typecheck/test/build for web) — no gap to fill. |
| Terraform scope | Artifact Registry, Cloud SQL (private-IP only), GCS + HMAC key, WIF pool/provider, runtime + CI service accounts, required-API enablement | Matches the task's explicit substrate list. GKE cluster/node-pool provisioning and Secret Manager *secret* resources are deliberately out of scope (see `infra/terraform/README.md` "Deliberately out of scope") — provisioning real compute/GPU quota and writing real secret values are operational steps, not something to plan blind in an agent-authored skeleton. |

## What's deliberately NOT done

- No real `terraform plan`/`apply`, no real `gcloud run deploy`, no real GKE cluster — all need
  live cloud credentials this sandbox doesn't have. Structural validation only (see report).
- No image signing (cosign) step — task didn't require it and it needs a KMS key/OIDC-signing
  setup of its own; noted as a natural follow-on to the SBOM/Trivy gate in `docs/deployment.md`.
- No Helm charts — task specified Kustomize for GKE; Helm was mentioned only in
  `docs/architecture.md`'s aspirational IaC line and Kustomize is the concrete ask here.

## Validation performed

See the final task report for exact commands/output: `.dockerignore`-scoped `docker build` of the
api-gateway image, `kubectl kustomize` (kustomize v5, bundled with kubectl) against
`infra/k8s/base` and both overlays, `docker compose config` against the prod overlay, YAML
parsing of both workflow files, `terraform fmt`/`validate`, and a full unchanged-application
`ruff`/`pytest` run.
