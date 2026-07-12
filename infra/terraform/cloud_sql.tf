# Cloud SQL for Postgres — the cloud equivalent of infra/docker/docker-compose.yml's
# `postgres` service (metadata store for anodyne_app + Temporal's `temporal` DB,
# see docs/dev-runbook.md). Application connectivity mirrors the on-prem
# non-superuser-role pattern: Cloud SQL's default `postgres` user is the
# migration-time superuser-equivalent; `anodyne_app` (created by the same
# infra/docker/postgres/init/01-app-role.sql-equivalent bootstrap SQL, run
# once via `gcloud sql connect` or a migration Job) is what the app connects
# as, so Postgres RLS is enforced identically to local/on-prem.
resource "random_password" "postgres_root" {
  length  = 32
  special = false
}

resource "google_sql_database_instance" "anodyne" {
  name             = "anodyne-${var.environment}"
  database_version = "POSTGRES_16"
  region           = var.region

  settings {
    tier              = var.cloud_sql_tier
    disk_size         = var.cloud_sql_disk_size_gb
    disk_autoresize   = true
    availability_type = var.environment == "prod" ? "REGIONAL" : "ZONAL"

    backup_configuration {
      enabled                        = true
      point_in_time_recovery_enabled = var.environment == "prod"
    }

    ip_configuration {
      # No public IP; api-gateway/generation-worker (Cloud Run) and GKE both
      # reach this over the Cloud SQL Auth Proxy sidecar (see
      # infra/cloudrun/*.yaml `run.googleapis.com/cloudsql-instances` and the
      # GKE `cloudsql-proxy` sidecar pattern) rather than a routable IP.
      ipv4_enabled = false
    }
  }

  deletion_protection = var.environment == "prod"
}

resource "google_sql_database" "anodyne" {
  name     = "anodyne"
  instance = google_sql_database_instance.anodyne.name
}

resource "google_sql_database" "temporal" {
  name     = "temporal"
  instance = google_sql_database_instance.anodyne.name
}

resource "google_sql_user" "postgres" {
  name     = "postgres"
  instance = google_sql_database_instance.anodyne.name
  password = random_password.postgres_root.result
}

output "cloud_sql_connection_name" {
  description = "Used by the Cloud SQL Auth Proxy / Cloud Run's cloudsql-instances annotation."
  value       = google_sql_database_instance.anodyne.connection_name
}

output "cloud_sql_root_password" {
  value     = random_password.postgres_root.result
  sensitive = true
}
