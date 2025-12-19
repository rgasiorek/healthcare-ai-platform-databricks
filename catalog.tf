# NOTE: Using Hive Metastore instead of Unity Catalog (no AWS S3 setup required)
# Unity Catalog requires external storage (S3) which needs AWS configuration
# For this demo, we use the default Hive metastore with DBFS storage

# DBFS directories for healthcare data organization
# These mirror the Bronze/Silver/Gold medallion architecture

# Create DBFS directory structure via notebook
resource "databricks_notebook" "setup_directories" {
  path     = "/Shared/setup-dbfs-directories"
  language = "PYTHON"
  content_base64 = base64encode(<<-EOT
    # Databricks notebook source
    # MAGIC %md
    # MAGIC # Setup DBFS Directory Structure
    # MAGIC
    # MAGIC Creates directory structure for healthcare data:
    # MAGIC - Bronze: Raw data from Kaggle
    # MAGIC - Silver: Cleaned/transformed data
    # MAGIC - Gold: Business-ready aggregations

    # COMMAND ----------
    # Create directory structure
    dbutils.fs.mkdirs("/healthcare/bronze/xray_images")
    dbutils.fs.mkdirs("/healthcare/silver/")
    dbutils.fs.mkdirs("/healthcare/gold/")
    dbutils.fs.mkdirs("/healthcare/models/")

    print("✅ Directory structure created:")
    print("   /healthcare/bronze/xray_images - Raw X-ray images")
    print("   /healthcare/silver/ - Cleaned data")
    print("   /healthcare/gold/ - Business-ready data")
    print("   /healthcare/models/ - ML models")

    # COMMAND ----------
    # Verify directories
    display(dbutils.fs.ls("/healthcare/"))
  EOT
  )
}

# Output storage paths
output "bronze_path" {
  value       = "/healthcare/bronze"
  description = "Path to bronze (raw) data storage"
}

output "xray_images_path" {
  value       = "/healthcare/bronze/xray_images"
  description = "Path to X-ray images storage"
}

output "silver_path" {
  value       = "/healthcare/silver"
  description = "Path to silver (cleaned) data storage"
}

output "gold_path" {
  value       = "/healthcare/gold"
  description = "Path to gold (business-ready) data storage"
}
