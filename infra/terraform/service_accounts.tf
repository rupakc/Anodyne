# Runtime service accounts — one per deployable, least-privilege, distinct
# from the CI/CD deploy identity in workload_identity.tf. Bound to Cloud Run
# (infra/cloudrun/*.yaml `serviceAccountName`) or to GKE Workload Identity
# (infra/k8s/base/*.yaml k8s ServiceAccounts, annotated to impersonate these
# — see the `google_service_account_iam_member` "workload identity user"
# bindings below).
resource "google_service_account" "api_gateway" {
  account_id   = "anodyne-gateway-${var.environment}"
  display_name = "Anodyne api-gateway runtime (${var.environment})"
}

resource "google_service_account" "generation_worker" {
  account_id   = "anodyne-worker-${var.environment}"
  display_name = "Anodyne generation-worker runtime (${var.environment})"
}

locals {
  cloud_sql_client_sas = [
    google_service_account.api_gateway.email,
    google_service_account.generation_worker.email,
  ]
}

resource "google_project_iam_member" "cloud_sql_client" {
  for_each = toset(local.cloud_sql_client_sas)
  project  = var.project_id
  role     = "roles/cloudsql.client"
  member   = "serviceAccount:${each.value}"
}

resource "google_project_iam_member" "secret_accessor" {
  for_each = toset(local.cloud_sql_client_sas)
  project  = var.project_id
  role     = "roles/secretmanager.secretAccessor"
  member   = "serviceAccount:${each.value}"
}

# GKE Workload Identity binding: lets the in-cluster Kubernetes
# ServiceAccounts (infra/k8s/base/api-gateway.yaml `anodyne-api-gateway`,
# infra/k8s/base/generation-worker.yaml `anodyne-generation-worker`)
# impersonate these GCP service accounts without any mounted key file.
# `k8s_namespace` defaults to the base overlay's namespace; override per
# environment (anodyne-dev / anodyne-prod) when applying per-overlay.
variable "k8s_namespace" {
  description = "Kubernetes namespace whose KSAs are bound to the runtime GSAs via Workload Identity."
  type        = string
  default     = "anodyne"
}

resource "google_service_account_iam_member" "gateway_workload_identity" {
  service_account_id = google_service_account.api_gateway.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/anodyne-api-gateway]"
}

resource "google_service_account_iam_member" "worker_workload_identity" {
  service_account_id = google_service_account.generation_worker.name
  role               = "roles/iam.workloadIdentityUser"
  member             = "serviceAccount:${var.project_id}.svc.id.goog[${var.k8s_namespace}/anodyne-generation-worker]"
}

output "api_gateway_service_account_email" {
  value = google_service_account.api_gateway.email
}

output "generation_worker_service_account_email" {
  value = google_service_account.generation_worker.email
}
