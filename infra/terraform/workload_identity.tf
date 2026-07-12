# Workload Identity Federation for GitHub Actions — the WIF pool + provider
# .github/workflows/build-and-push.yml authenticates against via
# `google-github-actions/auth`, `workload_identity_provider:` input. No
# Service Account JSON key is ever created or stored: GitHub's OIDC token is
# exchanged directly for short-lived GCP credentials, scoped down to the
# `attribute_condition` below.
resource "google_iam_workload_identity_pool" "github" {
  workload_identity_pool_id = "anodyne-github-pool"
  display_name              = "Anodyne GitHub Actions"
  description               = "Federates GitHub Actions OIDC tokens for CI/CD image publishing (no long-lived keys)."
}

resource "google_iam_workload_identity_pool_provider" "github" {
  workload_identity_pool_id          = google_iam_workload_identity_pool.github.workload_identity_pool_id
  workload_identity_pool_provider_id = "github-actions"
  display_name                       = "GitHub Actions OIDC"

  attribute_mapping = {
    "google.subject"       = "assertion.sub"
    "attribute.repository" = "assertion.repository"
    "attribute.ref"        = "assertion.ref"
  }

  # Narrows federation to exactly this repo — without this, ANY GitHub
  # Actions workflow in ANY repo whose OIDC token this pool trusts could
  # impersonate the deploy SA. This is the load-bearing security boundary of
  # WIF; do not widen it casually.
  attribute_condition = "assertion.repository == \"${var.github_repository}\""

  oidc {
    issuer_uri = "https://token.actions.githubusercontent.com"
  }
}

# CI/CD's own deploy identity — distinct from the api-gateway/generation-worker
# *runtime* service accounts (service_accounts.tf): this one only ever needs
# Artifact Registry write access, never database/secret access.
resource "google_service_account" "github_deploy" {
  account_id   = "anodyne-github-deploy"
  display_name = "Anodyne GitHub Actions deploy (build-and-push.yml)"
}

resource "google_project_iam_member" "github_deploy_ar_writer" {
  project = var.project_id
  role    = "roles/artifactregistry.writer"
  member  = "serviceAccount:${google_service_account.github_deploy.email}"
}

# Grants the WIF pool permission to impersonate github_deploy, restricted to
# the configured branches/tags via the pool provider's attribute_condition
# above plus this member's `attribute.ref` matrix.
resource "google_service_account_iam_member" "github_deploy_wif_binding" {
  for_each           = toset(var.github_deploy_branches)
  service_account_id = google_service_account.github_deploy.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "principalSet://iam.googleapis.com/${google_iam_workload_identity_pool.github.name}/attribute.ref/${each.value}"
}

output "workload_identity_provider" {
  description = "Full resource name for GitHub Actions `workload_identity_provider:` input (repo/org variable GCP_WORKLOAD_IDENTITY_PROVIDER)."
  value       = google_iam_workload_identity_pool_provider.github.name
}

output "github_deploy_service_account_email" {
  description = "GitHub Actions `service_account:` input (repo/org variable GCP_SERVICE_ACCOUNT)."
  value       = google_service_account.github_deploy.email
}
