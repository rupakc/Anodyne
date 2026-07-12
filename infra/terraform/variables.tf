variable "project_id" {
  description = "GCP project ID that hosts the Anodyne substrate."
  type        = string
}

variable "region" {
  description = "Primary GCP region for regional resources (Artifact Registry, Cloud SQL, Cloud Run)."
  type        = string
  default     = "us-central1"
}

variable "environment" {
  description = "Deployment environment name, used as a resource-naming suffix (dev|staging|prod)."
  type        = string
  default     = "dev"
  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "artifact_registry_repository_id" {
  description = "Artifact Registry Docker repository name (see .github/workflows/build-and-push.yml GAR_REPOSITORY)."
  type        = string
  default     = "anodyne"
}

variable "cloud_sql_tier" {
  description = "Cloud SQL machine tier for the Postgres instance backing anodyne_app + Temporal."
  type        = string
  default     = "db-custom-2-8192" # 2 vCPU / 8GB — dev-sized; raise for prod overlay
}

variable "cloud_sql_disk_size_gb" {
  description = "Cloud SQL disk size in GB."
  type        = number
  default     = 50
}

variable "gcs_bucket_name" {
  description = "Name of the GCS bucket used as the S3-compatible object store (ANODYNE_S3_BUCKET). Must be globally unique; defaults to a project-qualified name."
  type        = string
  default     = ""
}

variable "github_repository" {
  description = "GitHub \"owner/repo\" allowed to federate via Workload Identity (e.g. \"anodyne-org/anodyne\"). Scopes which repo's OIDC tokens the pool trusts — see workload_identity.tf."
  type        = string
}

variable "github_deploy_branches" {
  description = "Git refs (branches/tags) allowed to assume the deploy service account, matched against the OIDC token's `assertion.ref` claim. Keep narrow — this is the blast radius of a compromised workflow run."
  type        = list(string)
  default     = ["refs/heads/main"]
}

variable "labels" {
  description = "Common resource labels applied across the substrate."
  type        = map(string)
  default = {
    app       = "anodyne"
    managedby = "terraform"
  }
}
