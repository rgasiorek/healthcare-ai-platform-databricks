# Data processing cluster for healthcare workloads
# Note: This cluster is defined but will not auto-start
# Configured for Unity Catalog with single-user access mode
resource "databricks_cluster" "healthcare_compute" {
  cluster_name            = "healthcare-data-cluster-${var.environment}"
  spark_version           = "15.4.x-scala2.12"  # Latest LTS with Unity Catalog support
  node_type_id            = "i3.xlarge"
  autotermination_minutes = 20
  num_workers             = 2

  # Enable Unity Catalog
  data_security_mode = "SINGLE_USER"

  spark_conf = {
    "spark.databricks.delta.preview.enabled" = "true"
  }

  # AWS tags - adjust based on your organization's tag policy requirements
  custom_tags = {
    "Project"     = "healthcare-xray"
    "Environment" = var.environment
    "CostCenter"  = "data-engineering"
  }

  depends_on = [databricks_metastore_assignment.workspace]
}

# Output cluster information

