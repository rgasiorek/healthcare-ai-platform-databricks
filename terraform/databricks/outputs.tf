# Databricks Module Outputs

# Unity Catalog
output "metastore_id" {
  value       = databricks_metastore.healthcare.id
  description = "Unity Catalog Metastore ID"
}

output "catalog_name" {
  value       = databricks_catalog.healthcare.name
  description = "Healthcare catalog name"
}

# External Locations
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

# Volume
output "volume_path" {
  value       = "/Volumes/${databricks_catalog.healthcare.name}/${databricks_schema.bronze.name}/${databricks_volume.xray_images.name}"
  description = "Path to X-ray images volume"
}

# Tables
output "bronze_table_full_name" {
  value       = "${databricks_catalog.healthcare.name}.${databricks_schema.bronze.name}.${databricks_sql_table.bronze_kaggle_metadata.name}"
  description = "Full name of bronze kaggle metadata table"
}

output "silver_xray_table_full_name" {
  value       = "${databricks_catalog.healthcare.name}.${databricks_schema.silver.name}.${databricks_sql_table.silver_xray_metadata.name}"
  description = "Full name of silver X-ray metadata table"
}

output "gold_predictions_table_full_name" {
  value       = "${databricks_catalog.healthcare.name}.${databricks_schema.gold.name}.${databricks_sql_table.gold_predictions.name}"
  description = "Full name of gold predictions table"
}

# Compute
output "cluster_id" {
  value       = databricks_cluster.healthcare_compute.id
  description = "Healthcare compute cluster ID"
}

output "cluster_url" {
  value       = "https://dbc-68a1cdfa-43b8.cloud.databricks.com/#setting/clusters/${databricks_cluster.healthcare_compute.id}/configuration"
  description = "Direct URL to cluster configuration"
}

output "warehouse_id" {
  value       = databricks_sql_endpoint.healthcare_warehouse.id
  description = "SQL Warehouse ID"
}

output "warehouse_url" {
  value       = "https://dbc-68a1cdfa-43b8.cloud.databricks.com/sql/warehouses/${databricks_sql_endpoint.healthcare_warehouse.id}"
  description = "Direct URL to SQL Warehouse"
}

# Notebooks
output "hello_world_notebook_path" {
  value       = databricks_notebook.hello_world.path
  description = "Path to Hello World notebook"
}

output "ingestion_notebook_path" {
  value       = databricks_notebook.kaggle_ingestion.path
  description = "Path to Kaggle ingestion notebook"
}
