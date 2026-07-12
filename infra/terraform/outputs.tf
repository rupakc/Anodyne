# Cross-cutting outputs that don't belong to a single resource file. Per-
# resource outputs (Artifact Registry URL, Cloud SQL connection name, GCS
# bucket/HMAC, WIF provider/SA) live alongside their resources — see
# artifact_registry.tf, cloud_sql.tf, gcs.tf, workload_identity.tf.
output "project_id" {
  value = var.project_id
}

output "region" {
  value = var.region
}

output "environment" {
  value = var.environment
}
