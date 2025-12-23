# AWS Module Variables

variable "environment" {
  description = "Environment name (dev, pilot, prod)"
  type        = string
}

variable "databricks_account_id" {
  description = "Databricks Account ID for IAM trust policy ExternalId"
  type        = string
}

variable "aws_region" {
  description = "AWS region for resources"
  type        = string
}
