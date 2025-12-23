# Databricks Secrets - Kaggle API Credentials

# Secret Scope for Kaggle credentials
resource "databricks_secret_scope" "kaggle" {
  name = "kaggle"
}

# Store Kaggle username
resource "databricks_secret" "kaggle_username" {
  scope        = databricks_secret_scope.kaggle.name
  key          = "username"
  string_value = "radoslawgasiorek"
}

# Store Kaggle API token
resource "databricks_secret" "kaggle_token" {
  scope        = databricks_secret_scope.kaggle.name
  key          = "token"
  string_value = "KGAT_803bbdf77fd1f5aa63c2bafb28891739"
}
