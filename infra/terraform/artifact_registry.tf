# Artifact Registry Docker repository the build-and-push CI workflow
# (.github/workflows/build-and-push.yml) pushes api-gateway/generation-worker/
# web images into.
resource "google_artifact_registry_repository" "anodyne" {
  location      = var.region
  repository_id = var.artifact_registry_repository_id
  description   = "Anodyne container images (api-gateway, generation-worker, web)"
  format        = "DOCKER"
  labels        = var.labels

  docker_config {
    # CI's Trivy gate scans images pre-push; Artifact Registry's own
    # vulnerability scanning is a second, continuous layer for anything
    # that ships before a newly-disclosed CVE existed.
    immutable_tags = false
  }
}

output "artifact_registry_repository_url" {
  description = "Docker repository URL, e.g. for `docker push <url>/api-gateway:<tag>`."
  value       = "${var.region}-docker.pkg.dev/${var.project_id}/${google_artifact_registry_repository.anodyne.repository_id}"
}
