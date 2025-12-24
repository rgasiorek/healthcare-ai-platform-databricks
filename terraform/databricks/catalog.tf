# Unity Catalog Configuration with AWS S3 Storage
# IAM roles created successfully - now creating full Unity Catalog setup

# Data source for AWS region
data "aws_region" "current" {}

# Storage Credential for Unity Catalog Metastore
resource "databricks_storage_credential" "unity_catalog_metastore" {
  name = "unity-catalog-metastore-credential-${var.environment}"
  aws_iam_role {
    role_arn = var.metastore_iam_role_arn
  }
  comment = "Storage credential for Unity Catalog metastore S3 bucket (${var.environment})"
  depends_on = [databricks_metastore_assignment.workspace]
}

# Unity Catalog Metastore - NEW metastore with S3 storage
resource "databricks_metastore" "healthcare" {
  name          = "healthcare-metastore-${var.environment}-${data.aws_region.current.name}"
  storage_root  = "s3://${var.unity_catalog_bucket_id}/metastore"
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
    role_arn = var.data_access_iam_role_arn
  }
  comment = "Storage credential for healthcare data lake S3 bucket (${var.environment})"
  depends_on = [databricks_metastore_assignment.workspace]
}

# External Location for Bronze Layer (Raw Data)
resource "databricks_external_location" "bronze" {
  name            = "healthcare-bronze-${var.environment}"
  url             = "s3://${var.healthcare_data_bucket_id}/bronze"
  credential_name = databricks_storage_credential.healthcare_data.id
  comment         = "Bronze layer: Raw healthcare data from Kaggle (${var.environment})"
  depends_on      = [databricks_metastore_assignment.workspace]
}

# External Location for Silver Layer (Cleaned Data)
resource "databricks_external_location" "silver" {
  name            = "healthcare-silver-${var.environment}"
  url             = "s3://${var.healthcare_data_bucket_id}/silver"
  credential_name = databricks_storage_credential.healthcare_data.id
  comment         = "Silver layer: Cleaned and transformed healthcare data (${var.environment})"
  depends_on      = [databricks_metastore_assignment.workspace]
}

# External Location for Gold Layer (Business-Ready Data)
resource "databricks_external_location" "gold" {
  name            = "healthcare-gold-${var.environment}"
  url             = "s3://${var.healthcare_data_bucket_id}/gold"
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





