# Enables the GCP APIs every other .tf file in this skeleton depends on.
# Kept as its own resource set (rather than assumed pre-enabled) so a fresh
# project only needs `project_id` + `github_repository` set to apply cleanly.
locals {
  required_services = [
    "run.googleapis.com",              # Cloud Run (api-gateway, generation-worker)
    "artifactregistry.googleapis.com", # Artifact Registry (container images)
    "sqladmin.googleapis.com",         # Cloud SQL
    "storage.googleapis.com",          # GCS (object store)
    "iam.googleapis.com",              # Service accounts
    "iamcredentials.googleapis.com",   # WIF short-lived credential minting
    "sts.googleapis.com",              # WIF token exchange
    "secretmanager.googleapis.com",    # Secret Manager (env/secret refs)
    "container.googleapis.com",        # GKE (Kustomize deployment target)
    "cloudfunctions.googleapis.com",   # Cloud Functions (thin webhook receivers)
  ]
}

resource "google_project_service" "required" {
  for_each = toset(local.required_services)
  project  = var.project_id
  service  = each.value

  disable_dependent_services = false
  disable_on_destroy         = false
}
