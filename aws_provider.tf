# AWS Data Sources for Unity Catalog
# Gets current AWS account and region information

# Get current AWS account information
data "aws_caller_identity" "current" {}

data "aws_region" "current" {}

# Outputs for reference
output "aws_account_id" {
  value       = data.aws_caller_identity.current.account_id
  description = "AWS Account ID"
}

output "aws_region" {
  value       = data.aws_region.current.name
  description = "AWS Region"
}
