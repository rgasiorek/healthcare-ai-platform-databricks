# Healthcare AI Platform - Databricks on AWS
# Main Terraform Configuration

terraform {
  required_version = ">= 1.0"
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.50"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

# Provider Configurations
provider "databricks" {
  # Configuration will be read from ~/.databrickscfg
}

provider "aws" {
  region  = var.aws_region
  profile = "DevAdmin-905418100642"  # TODO: Make this configurable

  default_tags {
    tags = {
      Project     = "healthcare-ai"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# AWS Module - S3 buckets and IAM roles
module "aws" {
  source = "./aws"

  environment            = var.environment
  databricks_account_id  = var.databricks_account_id
  aws_region            = var.aws_region
}

# Databricks Module - Unity Catalog, tables, clusters, notebooks
module "databricks" {
  source = "./databricks"

  environment = var.environment

  # AWS resources from aws module
  unity_catalog_bucket_id        = module.aws.unity_catalog_bucket_id
  healthcare_data_bucket_id      = module.aws.healthcare_data_bucket_id
  metastore_iam_role_arn        = module.aws.metastore_iam_role_arn
  data_access_iam_role_arn      = module.aws.data_access_iam_role_arn

  # Module dependencies
  depends_on = [module.aws]
}
