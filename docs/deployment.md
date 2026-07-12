# Deployment & CI/CD

> Companion to `docs/architecture.md` §3 (service topology) and §5 (IaC/CI-CD). Covers the path
> from local dev → on-prem Docker Compose → GCP Cloud Run → GCP GKE, plus the CI/CD pipeline and
> the secret-management model that's consistent across all four. See
> `docs/superpowers/specs/2026-07-12-deploy-i-design.md` for the design rationale and trade-offs.

## The three deployable units

| Unit | Entrypoint | Dockerfile | Has an HTTP surface? |
|---|---|---|---|
| `api-gateway` | `uvicorn api_gateway.app:create_app --factory` | `apps/api-gateway/Dockerfile` | Yes — `:8080`, `/healthz` `/readyz` |
| `generation-worker` | `python -m generation_worker.main` | `apps/generation-worker/Dockerfile` | No — Temporal task-queue consumer |
| `web` | `node server.js` (Next.js standalone) | `apps/web/Dockerfile` | Yes — `:3000` |

All three build from the **repo root** as Docker context (`docker build -f apps/<x>/Dockerfile .`)
— the Python images need the full `uv` workspace on disk to resolve `[tool.uv.sources]` path
dependencies even though only one app's own dependency set is ultimately installed. See
`.dockerignore` for what's excluded from every build context (notably `.env*`, `.git`, docs, IaC).

## 1. Local dev

Unchanged — see `README.md` "Quick start" and `docs/dev-runbook.md`. `make up` runs the backbone
(Postgres/Redis/MinIO/Keycloak/Temporal/Ray/Ollama) via `infra/docker/docker-compose.yml`; the three
app processes run on the host via `make dev`, not in containers. This sub-system doesn't change
that flow — it adds the *next* step, containerizing those same three processes.

To build and smoke-test an image locally without any orchestration:

```bash
docker build -f apps/api-gateway/Dockerfile -t anodyne/api-gateway:local .
docker build -f apps/generation-worker/Dockerfile -t anodyne/generation-worker:local .
docker build -f apps/web/Dockerfile -t anodyne/web:local .
```

(`make docker-build` runs all three — see the Makefile.)

## 2. On-prem: Docker Compose (prod profile)

`infra/docker/docker-compose.prod.yml` **layers on top of** the dev backbone rather than
duplicating it:

```bash
docker compose -f infra/docker/docker-compose.yml \
                -f infra/docker/docker-compose.prod.yml \
                up -d --build
```

What the prod overlay changes:
- Adds the three app containers (built from the Dockerfiles above), each with `restart:
  unless-stopped`.
- Removes host port publishing for backbone services that shouldn't be reachable outside the
  Docker network in prod (Postgres, Redis, MinIO, Ray dashboard/client, Temporal UI). Only
  `api-gateway` (`:8000`) and `web` (`:3000`) stay published.
- Adds `deploy.resources.limits` (cpus/memory) throughout — Compose v2 honors these outside Swarm
  mode too.

**Known gap, on purpose:** Keycloak still runs `start-dev` (inherited unchanged from the dev
backbone). A real production Keycloak needs its own external database, `start` (not `start-dev`),
and TLS termination — that's a Keycloak-operations concern independent of Anodyne's own
deployment, and out of scope here. Put Keycloak behind the same reverse proxy as `web`/
`api-gateway` (see below) and follow Keycloak's own production guide before going live.

**Also out of scope, flagged:** no TLS-terminating reverse proxy (Caddy/nginx) is included; add
one in front of `api-gateway:8000` / `web:3000` / `keycloak:8080` for a real on-prem deployment.

Secrets: every credential referenced by the prod overlay comes from the environment (the same
`-include .env` / `export` pattern the root `Makefile` already uses) — never hardcoded, and
`.env*` files stay untouched by this sub-system and out of every Docker build context.

## 3. GCP: Cloud Run

For lighter-weight or tabular/text-only deployments that don't need a real Ray cluster or GPUs.

- `infra/cloudrun/api-gateway-service.yaml` — public, `minScale: 1` (holds the `/jobs/{id}/stream`
  WebSocket, so scale-to-zero would drop live connections), Cloud SQL via the Auth Proxy
  annotation, all secrets via `secretKeyRef`.
- `infra/cloudrun/generation-worker-service.yaml` — modeled as an **internal-ingress Cloud Run
  Service** (not a Job): it's a long-lived Temporal task-queue consumer, not request-driven or
  run-to-completion, so neither pure Cloud Run primitive fits exactly. `cpu-throttling: false` +
  `minScale: 1` keep it always-on and CPU-allocated between (non-existent) requests.

Both are Knative-shaped YAML with `${VAR}` placeholders, deployed via:

```bash
envsubst < infra/cloudrun/api-gateway-service.yaml | \
  gcloud run services replace - --region "$GCP_REGION" --project "$GCP_PROJECT_ID"
```

GCS as the object store: `packages/anodyne-storage`'s boto3-based `S3ObjectStore` talks to GCS via
its S3-compatible XML API (`AWS_ENDPOINT_URL_S3=https://storage.googleapis.com`) with HMAC keys
(provisioned in `infra/terraform/gcs.tf`) — the exact same client code path as MinIO on-prem, no
branching in application code.

When generation-worker outgrows Cloud Run (GPU-backed image/audio/video generation, real Ray
cluster scale), move it to GKE — see §4.

## 4. GCP: GKE via Kustomize

```
infra/k8s/
  base/            namespace, configmap, secret (placeholder), api-gateway.yaml,
                   generation-worker.yaml, web.yaml, raycluster.yaml (KubeRay, see below)
  overlays/dev/    namePrefix dev-, 1 replica each, dev hostnames
  overlays/prod/   namePrefix prod-, 3 replicas, wider HPA ceiling
```

```bash
kubectl kustomize infra/k8s/overlays/prod | envsubst | kubectl apply -f -
```

Same `envsubst` pattern as Cloud Run: `images[].newName` in each overlay's `kustomization.yaml`
contains literal `${GAR_LOCATION}`/`${GCP_PROJECT_ID}`/`${GAR_REPOSITORY}`/`${IMAGE_TAG}`
placeholders (Kustomize itself does no shell interpolation) — `kustomize build`/`kubectl
kustomize` still succeeds and was verified structurally (see report).

**Ray cluster:** `infra/k8s/base/raycluster.yaml` is a KubeRay `RayCluster` CRD manifest — the
GKE-scale replacement for the single-node `ray-head` container in the dev Compose backbone. It's
**not** included in any `kustomization.yaml` resource list because the KubeRay operator/CRDs are a
cluster-scoped, one-time Helm install (`kuberay-operator`), and `kustomize build` must succeed
against a cluster that doesn't have it yet. Once the operator is installed:

```bash
kubectl apply -f infra/k8s/base/raycluster.yaml -n anodyne
```

It ships a `cpu-workers` group (2 replicas) and a `gpu-workers` group (0 replicas by default,
tainted `nvidia.com/gpu`) — scale the latter up once real GPU node-pool quota exists for
self-hosted image/audio/video model inference.

**Security posture:** every container runs non-root, `readOnlyRootFilesystem: true`, all Linux
capabilities dropped, with an `emptyDir` mounted at `/tmp` (`HOME=/tmp`) for third-party scratch
use — Ray, and the torch-backed CTGAN/TVAE/SDXL/TTS plumbing pulled in transitively, write scratch
files even though no first-party Anodyne code under `packages/*/src` or `apps/*/src` does (verified
by grep — see design spec).

**Secrets:** `infra/k8s/base/secret.yaml` is a placeholder only (`anodyne.dev/placeholder: "true"`
annotation, `CHANGEME` values) — it exists so `kustomize build` renders a complete object graph. A
real cluster replaces it with either a SealedSecret / External Secrets Operator `ExternalSecret`
pulling from GCP Secret Manager, or an out-of-band `kubectl create secret generic` that's never
committed. **Never edit `infra/k8s/base/secret.yaml` to hold a real value.**

## 5. GCP substrate: Terraform

`infra/terraform/` is a **skeleton** provisioning Artifact Registry, Cloud SQL (Postgres, private
IP only), a GCS bucket + HMAC key, GitHub Actions Workload Identity Federation, and the runtime
service accounts Cloud Run/GKE Workload Identity bind to. See `infra/terraform/README.md` for the
full bootstrap sequence (state-backend chicken-and-egg, apply order, wiring outputs into GitHub
repo variables and Secret Manager). Deliberately out of scope: the GKE cluster resource itself and
real Secret Manager secret *values* — see that README's "Deliberately out of scope" section.

## 6. CI/CD

- **Quality gate** (`.github/workflows/ci.yml`, unmodified — already met the requirement before
  this sub-system): `uv sync` → `ruff check`/`format --check` → `mypy` → `pytest -m "not
  integration and not e2e"`, a separate `integration` job (testcontainers Postgres RLS + Ray +
  Temporal), and a `web` job (`pnpm lint`/`typecheck`/`test`/`build`).
- **Build, scan, push** (`.github/workflows/build-and-push.yml`, new): matrix-builds all three
  images on every PR (build + SBOM + Trivy gate, no push — safe for fork PRs, no cloud credentials
  needed), and on `main`/version tags additionally authenticates via GCP Workload Identity
  Federation (`google-github-actions/auth`, **no JSON key**) and pushes to Artifact Registry.
  - SBOM: `anchore/sbom-action` (syft), SPDX-JSON, uploaded as a build artifact regardless of scan
    outcome.
  - Scan: `aquasecurity/trivy-action`, `severity: HIGH,CRITICAL`, `exit-code: "1"` — a failing scan
    fails the job before the push step ever runs.
  - Permissions: `contents: read` at workflow level; `id-token: write` added only at the job level
    (needed for the OIDC token, and only exercised on `push`, never on PRs).
  - Concurrency: one in-flight build per ref, superseded runs cancelled.
  - Tags: `:<sha>` always; `:latest` on `main`; `:<tag-name>` on a `v*.*.*` tag.

### One-time GCP setup this workflow assumes

1. Apply `infra/terraform/` (or provision by hand): Artifact Registry repo, WIF pool/provider,
   `anodyne-github-deploy` service account with `roles/artifactregistry.writer`.
2. Set these **repository variables** (Settings → Secrets and variables → Actions → Variables —
   these are identifiers, not credentials, so `vars.*` not `secrets.*`):
   `GCP_PROJECT_ID`, `GCP_AR_LOCATION`, `GCP_AR_REPOSITORY`, `GCP_WORKLOAD_IDENTITY_PROVIDER`,
   `GCP_SERVICE_ACCOUNT`.
3. No further action — `build-and-push.yml` picks these up automatically.

## Secret-management model (consistent across all four targets)

| Target | Secret mechanism |
|---|---|
| Local dev | `.env` (untracked, from `.env.example`) |
| On-prem Compose | Host environment / untracked `.env`, injected via Compose `environment:` |
| Cloud Run | GCP Secret Manager, referenced via `secretKeyRef` in the service YAML |
| GKE | K8s `Secret` (placeholder committed; real values via SealedSecrets/External Secrets/`kubectl create secret` out-of-band) |
| CI/CD → GCP auth | Workload Identity Federation — **no stored credential at all**, short-lived OIDC-derived tokens only |

No secret value is ever hardcoded in a Dockerfile, workflow, manifest, or Terraform file in this
sub-system; every `${VAR}`/`secretKeyRef`/`valueFrom` is a reference resolved at deploy/runtime.

## What needs real cloud credentials to verify (not done here)

- `terraform plan`/`apply` against a real project (validated structurally with `terraform
  fmt`/`validate` only — see report).
- An actual `gcloud run services replace` / `kubectl apply` to a live Cloud Run project or GKE
  cluster.
- The `build-and-push.yml` push step (WIF auth + Artifact Registry push) — PR builds exercise
  everything up to and including the Trivy gate without needing this.
- End-to-end Cloud SQL Auth Proxy / GCS HMAC connectivity from a deployed container.
