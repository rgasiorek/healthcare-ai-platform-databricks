# Environment configuration
variable "environment" {
  description = "Environment name (dev, pilot, prod)"
  type        = string
  default     = "dev"
}

# AWS Region
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "eu-central-1"
}

# Databricks Account ID
variable "databricks_account_id" {
  description = "Databricks Account ID for IAM trust policy ExternalId"
  type        = string
  default     = "5ed6b530-dbbf-4911-99df-b16b7863f1ef"
}

# Model Serving Authentication (set via environment variables)
variable "databricks_workspace_host" {
  description = "Databricks workspace URL (e.g., https://dbc-xxx.cloud.databricks.com)"
  type        = string
}

variable "databricks_model_serving_token" {
  description = "Databricks PAT token for model serving Files API access"
  type        = string
  sensitive   = true
}
