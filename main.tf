# Hello World - Simple Databricks Notebook
resource "databricks_notebook" "hello_world" {
  path     = "/Shared/terraform-hello-world"
  language = "PYTHON"
  content_base64 = base64encode(<<-EOT
    # Databricks notebook source
    print("Hello World from Terraform!")
    print("This notebook was created using Infrastructure as Code")

    # Display some basic info
    spark.sql("SELECT 'Terraform + Databricks = Success!' as message").display()
  EOT
  )
}

# Output the notebook path
output "notebook_path" {
  value       = databricks_notebook.hello_world.path
  description = "Path to the created notebook"
}

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
