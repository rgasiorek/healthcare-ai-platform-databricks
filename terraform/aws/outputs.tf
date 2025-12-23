# AWS Module Outputs

# S3 Buckets
output "unity_catalog_bucket_id" {
  value       = aws_s3_bucket.unity_catalog_metastore.id
  description = "Unity Catalog metastore S3 bucket name"
}

output "unity_catalog_bucket_arn" {
  value       = aws_s3_bucket.unity_catalog_metastore.arn
  description = "Unity Catalog metastore S3 bucket ARN"
}

output "healthcare_data_bucket_id" {
  value       = aws_s3_bucket.healthcare_data.id
  description = "Healthcare data lake S3 bucket name"
}

output "healthcare_data_bucket_arn" {
  value       = aws_s3_bucket.healthcare_data.arn
  description = "Healthcare data lake S3 bucket ARN"
}

# IAM Roles
output "metastore_iam_role_arn" {
  value       = aws_iam_role.unity_catalog_metastore.arn
  description = "IAM role ARN for Unity Catalog metastore"
}

output "data_access_iam_role_arn" {
  value       = aws_iam_role.healthcare_data_access.arn
  description = "IAM role ARN for healthcare data access"
}

# AWS Account Info
output "aws_account_id" {
  value       = data.aws_caller_identity.current.account_id
  description = "AWS Account ID"
}

output "aws_region" {
  value       = data.aws_region.current.name
  description = "AWS Region"
}
