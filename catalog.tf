# Unity Catalog Configuration with AWS S3 Storage
# IAM roles created successfully - now creating full Unity Catalog setup

# Storage Credential for Unity Catalog Metastore
resource "databricks_storage_credential" "unity_catalog_metastore" {
  name = "unity-catalog-metastore-credential-${var.environment}"
  aws_iam_role {
    role_arn = aws_iam_role.unity_catalog_metastore.arn
  }
  comment = "Storage credential for Unity Catalog metastore S3 bucket (${var.environment})"
}

# Unity Catalog Metastore - NEW metastore with S3 storage
resource "databricks_metastore" "healthcare" {
  name          = "healthcare-metastore-${var.environment}-${data.aws_region.current.name}"
  storage_root  = "s3://${aws_s3_bucket.unity_catalog_metastore.id}/metastore"
  owner         = "account users"
  region        = data.aws_region.current.name
  force_destroy = true  # For dev environment
}

# Assign metastore to workspace
resource "databricks_metastore_assignment" "workspace" {
  metastore_id = databricks_metastore.healthcare.id
  workspace_id = "3720263199172428"  # Your workspace ID from earlier
}

# Storage Credential for Healthcare Data Lake
resource "databricks_storage_credential" "healthcare_data" {
  name = "healthcare-data-lake-credential-${var.environment}"
  aws_iam_role {
    role_arn = aws_iam_role.healthcare_data_access.arn
  }
  comment = "Storage credential for healthcare data lake S3 bucket (${var.environment})"
  depends_on = [databricks_metastore_assignment.workspace]
}

# External Location for Bronze Layer (Raw Data)
resource "databricks_external_location" "bronze" {
  name            = "healthcare-bronze-${var.environment}"
  url             = "s3://${aws_s3_bucket.healthcare_data.id}/bronze"
  credential_name = databricks_storage_credential.healthcare_data.id
  comment         = "Bronze layer: Raw healthcare data from Kaggle (${var.environment})"
  depends_on      = [databricks_metastore_assignment.workspace]
}

# External Location for Silver Layer (Cleaned Data)
resource "databricks_external_location" "silver" {
  name            = "healthcare-silver-${var.environment}"
  url             = "s3://${aws_s3_bucket.healthcare_data.id}/silver"
  credential_name = databricks_storage_credential.healthcare_data.id
  comment         = "Silver layer: Cleaned and transformed healthcare data (${var.environment})"
  depends_on      = [databricks_metastore_assignment.workspace]
}

# External Location for Gold Layer (Business-Ready Data)
resource "databricks_external_location" "gold" {
  name            = "healthcare-gold-${var.environment}"
  url             = "s3://${aws_s3_bucket.healthcare_data.id}/gold"
  credential_name = databricks_storage_credential.healthcare_data.id
  comment         = "Gold layer: Business-ready healthcare analytics (${var.environment})"
  depends_on      = [databricks_metastore_assignment.workspace]
}

# Healthcare Catalog
resource "databricks_catalog" "healthcare" {
  metastore_id    = databricks_metastore.healthcare.id
  name            = "healthcare_catalog_${var.environment}"
  comment         = "Healthcare X-ray pneumonia analysis catalog with Unity Catalog governance (${var.environment})"
  properties = {
    purpose     = "healthcare-ai"
    domain      = "medical-imaging"
    environment = var.environment
  }
  depends_on = [databricks_metastore_assignment.workspace]
}

# Bronze Schema - Raw data layer
resource "databricks_schema" "bronze" {
  catalog_name = databricks_catalog.healthcare.name
  name         = "bronze"
  comment      = "Bronze layer: Raw data as ingested from Kaggle, immutable"
  properties = {
    layer   = "bronze"
    quality = "raw"
  }
}

# Silver Schema - Cleaned/transformed data
resource "databricks_schema" "silver" {
  catalog_name = databricks_catalog.healthcare.name
  name         = "silver"
  comment      = "Silver layer: Cleaned, validated, and enriched data"
  properties = {
    layer   = "silver"
    quality = "cleaned"
  }
}

# Gold Schema - Business-ready aggregated data
resource "databricks_schema" "gold" {
  catalog_name = databricks_catalog.healthcare.name
  name         = "gold"
  comment      = "Gold layer: Business-ready datasets, ML outputs, and aggregations"
  properties = {
    layer   = "gold"
    quality = "curated"
  }
}

# Volume for X-ray images in Bronze layer
resource "databricks_volume" "xray_images" {
  catalog_name = databricks_catalog.healthcare.name
  schema_name  = databricks_schema.bronze.name
  name         = "xray_images"
  volume_type  = "EXTERNAL"
  storage_location = "${databricks_external_location.bronze.url}/xray_images"
  comment      = "External volume for raw X-ray image files from Kaggle"
}

# Outputs
output "metastore_id" {
  value       = databricks_metastore.healthcare.id
  description = "Unity Catalog Metastore ID"
}

output "catalog_name" {
  value       = databricks_catalog.healthcare.name
  description = "Healthcare catalog name"
}

output "volume_path" {
  value       = "/Volumes/${databricks_catalog.healthcare.name}/${databricks_schema.bronze.name}/${databricks_volume.xray_images.name}"
  description = "Path to X-ray images volume"
}

output "bronze_external_location" {
  value       = databricks_external_location.bronze.url
  description = "S3 URL for Bronze layer"
}

output "silver_external_location" {
  value       = databricks_external_location.silver.url
  description = "S3 URL for Silver layer"
}

output "gold_external_location" {
  value       = databricks_external_location.gold.url
  description = "S3 URL for Gold layer"
}
