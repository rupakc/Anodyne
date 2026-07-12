# GCS bucket used as the S3-compatible object store in the cloud (see
# docs/architecture.md §2 — "Object storage: S3-compatible (GCS in cloud,
# MinIO on-prem)"). Application code talks to it via boto3's S3 client
# pointed at GCS's XML API (AWS_ENDPOINT_URL_S3=https://storage.googleapis.com),
# authenticated with HMAC keys (google_storage_hmac_key below) rather than
# native GCP IAM, so the exact same client code path runs against MinIO
# on-prem and GCS in the cloud with zero branching.
resource "random_id" "bucket_suffix" {
  byte_length = 4
}

resource "google_storage_bucket" "anodyne" {
  name          = var.gcs_bucket_name != "" ? var.gcs_bucket_name : "anodyne-${var.project_id}-${random_id.bucket_suffix.hex}"
  location      = var.region
  force_destroy = var.environment != "prod"
  labels        = var.labels

  uniform_bucket_level_access = true

  versioning {
    enabled = var.environment == "prod"
  }

  lifecycle_rule {
    condition {
      age = 90
    }
    action {
      type          = "SetStorageClass"
      storage_class = "NEARLINE"
    }
  }
}

# Dedicated service account for HMAC key issuance — narrower than binding
# HMAC to a human or the runtime SAs directly, so bucket key rotation is a
# single-resource operation.
resource "google_service_account" "gcs_hmac" {
  account_id   = "anodyne-gcs-hmac-${var.environment}"
  display_name = "Anodyne GCS HMAC key holder (${var.environment})"
}

resource "google_storage_hmac_key" "anodyne" {
  service_account_email = google_service_account.gcs_hmac.email
}

resource "google_storage_bucket_iam_member" "hmac_object_admin" {
  bucket = google_storage_bucket.anodyne.name
  role   = "roles/storage.objectAdmin"
  member = "serviceAccount:${google_service_account.gcs_hmac.email}"
}

output "gcs_bucket_name" {
  value = google_storage_bucket.anodyne.name
}

# Sensitive: wire into Secret Manager (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY),
# never into a CI log or a plain `terraform output`.
output "gcs_hmac_access_id" {
  value     = google_storage_hmac_key.anodyne.access_id
  sensitive = true
}

output "gcs_hmac_secret" {
  value     = google_storage_hmac_key.anodyne.secret
  sensitive = true
}
