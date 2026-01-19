# Databricks Jobs - Automated Workflows
# Jobs provide production-ready automation for notebook execution

# Job: Kaggle X-ray Data Ingestion
# Ingests chest X-ray images and metadata into Unity Catalog
resource "databricks_job" "kaggle_xray_ingestion" {
  name = "Kaggle X-ray Ingestion - ${var.environment}"

  description = <<-EOT
    Automated ingestion of Kaggle Chest X-ray dataset into Unity Catalog.

    This job:
    1. Downloads X-ray images from Kaggle dataset
    2. Uploads to Unity Catalog Volume (bronze.xray_images)
    3. Creates metadata records in bronze.kaggle_xray_metadata
    4. Validates data quality and completeness

    **Note**: This is typically a one-time or infrequent run.
    Data is persisted in Unity Catalog once ingested.
  EOT

  # Task: Run the ingestion notebook
  task {
    task_key = "ingest_kaggle_data"

    # Use the ingestion notebook
    notebook_task {
      notebook_path = databricks_notebook.ingest_kaggle_data.path
      source        = "WORKSPACE"
    }

    # Job cluster configuration (ephemeral, cost-effective)
    new_cluster {
      spark_version      = "15.4.x-scala2.12"  # Latest LTS
      node_type_id       = "i3.xlarge"
      num_workers        = 0  # Single-node (sufficient for data ingestion)

      # Enable Unity Catalog
      data_security_mode = "SINGLE_USER"

      spark_conf = {
        "spark.databricks.delta.preview.enabled" = "true"
      }

      custom_tags = {
        "Project"     = "healthcare-ai"
        "Environment" = var.environment
        "JobType"     = "data-ingestion"
      }
    }

    # Retry configuration
    max_retries               = 1
    timeout_seconds           = 3600  # 1 hour max
    min_retry_interval_millis = 60000  # Wait 1 min before retry
  }

  # Tags for organization
  tags = {
    type        = "ingestion"
    dataset     = "kaggle-xray"
    environment = var.environment
  }

  # Dependencies
  depends_on = [
    databricks_notebook.ingest_kaggle_data,
    databricks_catalog.healthcare,
    databricks_schema.bronze,
    databricks_volume.xray_images
  ]
}

# Output job information
output "ingestion_job_id" {
  description = "ID of the Kaggle X-ray ingestion job"
  value       = databricks_job.kaggle_xray_ingestion.id
}

output "ingestion_job_url" {
  description = "URL to view the ingestion job in Databricks workspace"
  value       = databricks_job.kaggle_xray_ingestion.url
}
