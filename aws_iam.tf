# IAM Role and Policies for Databricks Unity Catalog Access to S3
# This creates a cross-account IAM role that Databricks can assume

# Data source for Databricks account (you'll need to provide this)
# Get this from: Databricks Console → Account Settings → Account ID
variable "databricks_account_id" {
  description = "Databricks Account ID (from Account Console)"
  type        = string
  default     = "5ed6b530-dbbf-4911-99df-b16b7863f1ef"  # From your earlier workspace URL
}

# IAM Policy Document: Trust relationship allowing Databricks to assume this role
# AND allowing the role to assume itself (required for Unity Catalog)
data "aws_iam_policy_document" "databricks_assume_role" {
  statement {
    sid     = "DatabricksAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = ["arn:aws:iam::414351767826:role/unity-catalog-prod-UCMasterRole-14S5ZJVKOTYTL"]
    }

    condition {
      test     = "StringEquals"
      variable = "sts:ExternalId"
      values   = [var.databricks_account_id]
    }
  }

  # Allow the role to assume itself (self-assuming) - required for Unity Catalog
  statement {
    sid     = "SelfAssumeRole"
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "AWS"
      identifiers = [
        "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/databricks-unity-catalog-metastore-${var.environment}",
        "arn:aws:iam::${data.aws_caller_identity.current.account_id}:role/databricks-healthcare-data-access-${var.environment}"
      ]
    }
  }
}

# IAM Role for Unity Catalog Metastore
resource "aws_iam_role" "unity_catalog_metastore" {
  name               = "databricks-unity-catalog-metastore-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.databricks_assume_role.json

  tags = {
    Name        = "Databricks Unity Catalog Metastore Role"
    Description = "IAM role for Databricks Unity Catalog to access metastore S3 bucket"
    Environment = var.environment
  }
}

# IAM Policy: Permissions for Unity Catalog Metastore bucket
data "aws_iam_policy_document" "unity_catalog_metastore_s3" {
  statement {
    sid    = "MetastoreBucketAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
      "s3:GetLifecycleConfiguration",
      "s3:PutLifecycleConfiguration"
    ]
    resources = [
      aws_s3_bucket.unity_catalog_metastore.arn,
      "${aws_s3_bucket.unity_catalog_metastore.arn}/*"
    ]
  }

  statement {
    sid    = "MetastoreBucketList"
    effect = "Allow"
    actions = [
      "s3:ListBucket"
    ]
    resources = [
      aws_s3_bucket.unity_catalog_metastore.arn
    ]
  }
}

# Attach policy to metastore role
resource "aws_iam_role_policy" "unity_catalog_metastore_s3" {
  name   = "unity-catalog-metastore-s3-access"
  role   = aws_iam_role.unity_catalog_metastore.id
  policy = data.aws_iam_policy_document.unity_catalog_metastore_s3.json
}

# IAM Role for Healthcare Data Lake Access
resource "aws_iam_role" "healthcare_data_access" {
  name               = "databricks-healthcare-data-access-${var.environment}"
  assume_role_policy = data.aws_iam_policy_document.databricks_assume_role.json

  tags = {
    Name        = "Databricks Healthcare Data Access Role"
    Description = "IAM role for Databricks to access healthcare data lake"
    Environment = var.environment
  }
}

# IAM Policy: Permissions for Healthcare Data bucket
data "aws_iam_policy_document" "healthcare_data_s3" {
  statement {
    sid    = "HealthcareDataBucketAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation"
    ]
    resources = [
      aws_s3_bucket.healthcare_data.arn,
      "${aws_s3_bucket.healthcare_data.arn}/*"
    ]
  }
}

# Attach policy to data access role
resource "aws_iam_role_policy" "healthcare_data_s3" {
  name   = "healthcare-data-s3-access"
  role   = aws_iam_role.healthcare_data_access.id
  policy = data.aws_iam_policy_document.healthcare_data_s3.json
}

# Outputs
output "metastore_iam_role_arn" {
  value       = aws_iam_role.unity_catalog_metastore.arn
  description = "IAM Role ARN for Unity Catalog Metastore"
}

output "data_access_iam_role_arn" {
  value       = aws_iam_role.healthcare_data_access.arn
  description = "IAM Role ARN for Healthcare Data Access"
}
