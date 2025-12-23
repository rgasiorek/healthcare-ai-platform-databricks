# Top-Level Outputs - Exposing key outputs from AWS and Databricks modules

# AWS Outputs
output "aws_account_id" {
  value       = module.aws.aws_account_id
  description = "AWS Account ID"
}

output "aws_region" {
  value       = module.aws.aws_region
  description = "AWS Region"
}

output "unity_catalog_bucket_name" {
  value       = module.aws.unity_catalog_bucket_id
  description = "Unity Catalog metastore S3 bucket name"
}

output "healthcare_data_bucket_name" {
  value       = module.aws.healthcare_data_bucket_id
  description = "Healthcare data lake S3 bucket name"
}

output "metastore_iam_role_arn" {
  value       = module.aws.metastore_iam_role_arn
  description = "IAM Role ARN for Unity Catalog Metastore"
}

output "data_access_iam_role_arn" {
  value       = module.aws.data_access_iam_role_arn
  description = "IAM Role ARN for Healthcare Data Access"
}

# Databricks Outputs
output "metastore_id" {
  value       = module.databricks.metastore_id
  description = "Unity Catalog Metastore ID"
}

output "catalog_name" {
  value       = module.databricks.catalog_name
  description = "Healthcare catalog name"
}

output "bronze_external_location" {
  value       = module.databricks.bronze_external_location
  description = "S3 URL for Bronze layer"
}

output "silver_external_location" {
  value       = module.databricks.silver_external_location
  description = "S3 URL for Silver layer"
}

output "gold_external_location" {
  value       = module.databricks.gold_external_location
  description = "S3 URL for Gold layer"
}

output "volume_path" {
  value       = module.databricks.volume_path
  description = "Path to X-ray images volume"
}

output "bronze_table_full_name" {
  value       = module.databricks.bronze_table_full_name
  description = "Full name of bronze kaggle metadata table"
}

output "silver_xray_table_full_name" {
  value       = module.databricks.silver_xray_table_full_name
  description = "Full name of silver X-ray metadata table"
}

output "gold_predictions_table_full_name" {
  value       = module.databricks.gold_predictions_table_full_name
  description = "Full name of gold predictions table"
}

output "cluster_id" {
  value       = module.databricks.cluster_id
  description = "Healthcare compute cluster ID"
}

output "cluster_url" {
  value       = module.databricks.cluster_url
  description = "Direct URL to cluster configuration"
}

output "warehouse_id" {
  value       = module.databricks.warehouse_id
  description = "SQL Warehouse ID"
}

output "warehouse_url" {
  value       = module.databricks.warehouse_url
  description = "Direct URL to SQL Warehouse"
}

output "ingestion_notebook_path" {
  value       = module.databricks.ingestion_notebook_path
  description = "Path to Kaggle ingestion notebook"
}
