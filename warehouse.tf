# Serverless SQL Warehouse for cost-effective SQL queries
# Pay per query execution, not per hour - much cheaper for SQL workloads
resource "databricks_sql_endpoint" "healthcare_warehouse" {
  name             = "healthcare-sql-warehouse-${var.environment}"
  cluster_size     = "2X-Small"
  auto_stop_mins   = 10
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
output "warehouse_id" {
  value       = databricks_sql_endpoint.healthcare_warehouse.id
  description = "Serverless SQL Warehouse ID"
}

output "warehouse_url" {
  value       = "https://dbc-68a1cdfa-43b8.cloud.databricks.com/sql/warehouses/${databricks_sql_endpoint.healthcare_warehouse.id}"
  description = "Direct URL to SQL warehouse"
}

output "warehouse_jdbc_url" {
  value       = databricks_sql_endpoint.healthcare_warehouse.jdbc_url
  description = "JDBC connection string for SQL warehouse"
  sensitive   = true
}
