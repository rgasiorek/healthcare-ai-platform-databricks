# S3 Bucket for Unity Catalog Metastore Storage
# This bucket will store all Delta Lake tables managed by Unity Catalog

resource "aws_s3_bucket" "unity_catalog_metastore" {
  bucket = "healthcare-databricks-unity-catalog-${var.environment}"

  tags = {
    Name        = "Unity Catalog Metastore"
    Purpose     = "unity-catalog-storage"
    Environment = var.environment
  }
}

# Enable versioning for data protection
resource "aws_s3_bucket_versioning" "unity_catalog_metastore" {
  bucket = aws_s3_bucket.unity_catalog_metastore.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable server-side encryption
resource "aws_s3_bucket_server_side_encryption_configuration" "unity_catalog_metastore" {
  bucket = aws_s3_bucket.unity_catalog_metastore.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access
resource "aws_s3_bucket_public_access_block" "unity_catalog_metastore" {
  bucket = aws_s3_bucket.unity_catalog_metastore.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle policy to manage costs
resource "aws_s3_bucket_lifecycle_configuration" "unity_catalog_metastore" {
  bucket = aws_s3_bucket.unity_catalog_metastore.id

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  rule {
    id     = "transition-old-versions"
    status = "Enabled"

    filter {}

    noncurrent_version_transition {
      noncurrent_days = 30
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = 90
      storage_class   = "GLACIER"
    }

    noncurrent_version_expiration {
      noncurrent_days = 365
    }
  }
}

# S3 bucket for external data (Bronze/Silver/Gold layers)
resource "aws_s3_bucket" "healthcare_data" {
  bucket = "healthcare-data-lake-${var.environment}"

  tags = {
    Name        = "Healthcare Data Lake"
    Purpose     = "data-lake-storage"
    Environment = var.environment
  }
}

# Enable versioning for data lake
resource "aws_s3_bucket_versioning" "healthcare_data" {
  bucket = aws_s3_bucket.healthcare_data.id

  versioning_configuration {
    status = "Enabled"
  }
}

# Enable encryption for data lake
resource "aws_s3_bucket_server_side_encryption_configuration" "healthcare_data" {
  bucket = aws_s3_bucket.healthcare_data.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Block public access for data lake
resource "aws_s3_bucket_public_access_block" "healthcare_data" {
  bucket = aws_s3_bucket.healthcare_data.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Outputs
output "unity_catalog_bucket_name" {
  value       = aws_s3_bucket.unity_catalog_metastore.id
  description = "S3 bucket name for Unity Catalog metastore"
}

output "unity_catalog_bucket_arn" {
  value       = aws_s3_bucket.unity_catalog_metastore.arn
  description = "S3 bucket ARN for Unity Catalog metastore"
}

output "healthcare_data_bucket_name" {
  value       = aws_s3_bucket.healthcare_data.id
  description = "S3 bucket name for healthcare data lake"
}

output "healthcare_data_bucket_arn" {
  value       = aws_s3_bucket.healthcare_data.arn
  description = "S3 bucket ARN for healthcare data lake"
}
