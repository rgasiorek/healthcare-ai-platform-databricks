# Databricks Module Variables

variable "environment" {
  description = "Environment name (dev, pilot, prod)"
  type        = string
}

# AWS Resources (passed from aws module)
variable "unity_catalog_bucket_id" {
  description = "S3 bucket ID for Unity Catalog metastore"
  type        = string
}

variable "healthcare_data_bucket_id" {
  description = "S3 bucket ID for healthcare data lake"
  type        = string
}

variable "metastore_iam_role_arn" {
  description = "IAM role ARN for Unity Catalog metastore"
  type        = string
}

variable "data_access_iam_role_arn" {
  description = "IAM role ARN for healthcare data access"
  type        = string
}
