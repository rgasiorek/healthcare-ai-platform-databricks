# Databricks Apps - Feedback Review App Service Principal Permissions
# Service principal is auto-created by Databricks Apps when app is created
# This file manages the permissions for the service principal

# Note: The service principal itself is created automatically by Databricks Apps
# Service Principal Name: app-khxa2c radiologist-feedback-review
# Service Principal ID: f2a69ee5-e2d8-490b-a551-fa91155e53ed
# This is the application_id field that matches the app's ID

# Warehouse Permissions - Allow app to use SQL warehouse
resource "databricks_permissions" "feedback_app_warehouse" {
  sql_endpoint_id = databricks_sql_endpoint.healthcare_warehouse.id

  access_control {
    service_principal_name = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    permission_level       = "CAN_USE"
  }
}

# Catalog Grants - Allow app to use catalog
resource "databricks_grants" "feedback_app_catalog" {
  catalog = databricks_catalog.healthcare.name

  grant {
    principal  = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    privileges = ["USE_CATALOG"]
  }
}

# Gold Schema Grants - Allow app to use schema and select tables
resource "databricks_grants" "feedback_app_gold_schema" {
  schema = "${databricks_catalog.healthcare.name}.${databricks_schema.gold.name}"

  grant {
    principal  = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    privileges = ["USE_SCHEMA", "SELECT"]
  }
}

# Bronze Schema Grants - Allow app to use schema and select tables
resource "databricks_grants" "feedback_app_bronze_schema" {
  schema = "${databricks_catalog.healthcare.name}.${databricks_schema.bronze.name}"

  grant {
    principal  = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    privileges = ["USE_SCHEMA", "SELECT"]
  }
}

# Prediction Feedback Table Grants - Allow app to SELECT and MODIFY (INSERT)
resource "databricks_grants" "feedback_app_prediction_feedback_table" {
  table = "${databricks_catalog.healthcare.name}.${databricks_schema.gold.name}.${databricks_sql_table.gold_prediction_feedback.name}"

  grant {
    principal  = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    privileges = ["SELECT", "MODIFY"]
  }
}

# Gold Predictions Table Grants - Allow app to SELECT
resource "databricks_grants" "feedback_app_gold_predictions_table" {
  table = "${databricks_catalog.healthcare.name}.${databricks_schema.gold.name}.${databricks_sql_table.gold_predictions.name}"

  grant {
    principal  = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    privileges = ["SELECT"]
  }
}

# Bronze Kaggle Metadata Table Grants - Allow app to SELECT (for image paths)
resource "databricks_grants" "feedback_app_bronze_metadata_table" {
  table = "${databricks_catalog.healthcare.name}.${databricks_schema.bronze.name}.${databricks_sql_table.bronze_kaggle_metadata.name}"

  grant {
    principal  = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    privileges = ["SELECT"]
  }
}

# Volume Grants - Allow app to READ xray_images volume
resource "databricks_grants" "feedback_app_xray_volume" {
  volume = "${databricks_catalog.healthcare.name}.${databricks_schema.bronze.name}.${databricks_volume.xray_images.name}"

  grant {
    principal  = "f2a69ee5-e2d8-490b-a551-fa91155e53ed"
    privileges = ["READ_VOLUME"]
  }
}
