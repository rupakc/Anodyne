terraform {
  required_version = ">= 1.7.0"

  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 6.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  # Skeleton ships without a configured backend so `terraform validate`
  # doesn't require cloud credentials or a pre-existing bucket. Real usage
  # should configure a GCS backend, e.g.:
  #
  #   terraform {
  #     backend "gcs" {
  #       bucket = "anodyne-tfstate-<project-id>"
  #       prefix = "deploy-i"
  #     }
  #   }
  #
  # See README.md in this directory for the bootstrap sequence (the state
  # bucket itself can't be created by the config that also depends on it).
}

provider "google" {
  project = var.project_id
  region  = var.region
}
