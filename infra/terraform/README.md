# Anodyne GCP substrate — Terraform skeleton

Provisions the GCP-side platform substrate: Artifact Registry, Cloud SQL
(Postgres), a GCS bucket (S3-compatible object store), Workload Identity
Federation for GitHub Actions (no long-lived JSON keys), and the runtime
service accounts Cloud Run / GKE Workload Identity bind to.

This is a **skeleton** — reviewed for well-formed, `terraform validate`-clean
HCL, but **not applied against a real project as part of this sub-system**
(no cloud credentials in this environment; see docs/deployment.md "What
needs real cloud credentials to verify"). Review resource sizing (`cloud_sql_tier`,
disk sizes) and IAM scope before applying to a real environment.

## Layout

| File | Provisions |
|---|---|
| `versions.tf` | Provider/version pins; backend left unconfigured (see below) |
| `variables.tf` | All inputs, incl. `project_id`, `github_repository` |
| `main.tf` | Enables required GCP APIs |
| `artifact_registry.tf` | Docker repo `.github/workflows/build-and-push.yml` pushes into |
| `cloud_sql.tf` | Postgres instance + `anodyne`/`temporal` databases (private IP only) |
| `gcs.tf` | Object-store bucket + HMAC key (boto3-compatible, mirrors MinIO on-prem) |
| `service_accounts.tf` | Runtime SAs for api-gateway/generation-worker + IAM bindings |
| `workload_identity.tf` | GitHub Actions OIDC pool/provider + scoped deploy SA |
| `outputs.tf` | Cross-cutting outputs |

## Bootstrap sequence

1. **State backend first.** This config ships without a configured backend
   (`versions.tf`) because the GCS bucket that would hold Terraform state
   can't be a resource *of* the config that depends on it. Create it once,
   by hand or a separate tiny config:
   ```
   gsutil mb -l <region> gs://anodyne-tfstate-<project-id>
   gsutil versioning set on gs://anodyne-tfstate-<project-id>
   ```
   Then uncomment the `backend "gcs" {}` block in `versions.tf`.
2. `terraform init`
3. `terraform plan -var project_id=<id> -var github_repository=<owner>/<repo>`
4. `terraform apply` (review the plan — Cloud SQL, GKE, and Artifact
   Registry all cost money as soon as they exist).
5. Take `workload_identity_provider` + `github_deploy_service_account_email`
   from the outputs and set them as **repository variables** (not secrets —
   they're identifiers, not credentials) `GCP_WORKLOAD_IDENTITY_PROVIDER` /
   `GCP_SERVICE_ACCOUNT`, plus `GCP_PROJECT_ID` / `GCP_AR_LOCATION` /
   `GCP_AR_REPOSITORY`, so `.github/workflows/build-and-push.yml` can push.
6. Take `gcs_hmac_access_id` / `gcs_hmac_secret` (sensitive outputs) and
   store them in Secret Manager — reference them from
   `infra/cloudrun/*.yaml` / `infra/k8s/base/secret.yaml`, never from a
   plain `terraform output` in CI logs.

## Validation performed (this sub-system)

- `terraform fmt -check` / `terraform validate` — see docs/deployment.md for
  the recorded result; both require no cloud credentials and were run
  in-sandbox.
- `terraform plan` was **not** run (needs real `project_id` + credentials).

## Deliberately out of scope here

- GKE cluster resource itself (the Kustomize manifests in `infra/k8s/`
  target *a* GKE cluster; provisioning the cluster/node-pools/GPU pool is a
  natural Terraform follow-up once real GPU quota is requested — see the
  `raycluster.yaml` GPU worker group comment).
- Secret Manager secret *resources* (only IAM access to them is granted here
  via `roles/secretmanager.secretAccessor`); creating the actual secret
  entries is an operational step (see docs/deployment.md) so their values
  are never proposed/planned by an agent.
