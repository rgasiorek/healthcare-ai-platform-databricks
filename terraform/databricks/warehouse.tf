# Serverless SQL Warehouse for cost-effective SQL queries
# Pay per query execution, not per hour - much cheaper for SQL workloads
resource "databricks_sql_endpoint" "healthcare_warehouse" {
  name                      = "healthcare-sql-warehouse-${var.environment}"
  cluster_size              = "2X-Small"
  auto_stop_mins            = 10
  enable_serverless_compute = true

  tags {
    custom_tags {
      key   = "Project"
      value = "healthcare-xray"
    }
    custom_tags {
      key   = "Environment"
      value = var.environment
    }
    custom_tags {
      key   = "CostCenter"
      value = "data-engineering"
    }
  }
}

# Output warehouse information


