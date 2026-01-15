# Lakeview/AI/BI Dashboard - Model Performance Monitoring
# Uses new databricks_dashboard resource (recommended by Databricks as of April 2025)

# Data source for current user
data "databricks_current_user" "me" {}

resource "databricks_dashboard" "model_performance" {
  display_name = "Model Performance Comparison"
  warehouse_id = databricks_sql_endpoint.healthcare_warehouse.id
  parent_path  = "/Workspace/Shared"

  # Load dashboard definition from JSON file
  serialized_dashboard = file("${path.module}/dashboard_definition.json")

  # Ignore changes to allow manual edits in UI
  lifecycle {
    ignore_changes = [
      serialized_dashboard
    ]
  }
}

# Output dashboard URL
output "lakeview_dashboard_url" {
  value       = "https://${data.databricks_current_user.me.workspace_url}/dashboards/${databricks_dashboard.model_performance.id}"
  description = "URL to access the Lakeview Model Performance Dashboard"
}
