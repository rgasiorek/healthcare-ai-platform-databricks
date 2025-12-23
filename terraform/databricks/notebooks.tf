# Databricks Notebooks - Deployed via Terraform
# Notebook source code stored in /notebooks/ directory

# Hello World Notebook
resource "databricks_notebook" "hello_world" {
  path     = "/Shared/terraform-hello-world"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/00_utils/hello_world.py"
}

# Kaggle X-ray Data Ingestion Notebook
resource "databricks_notebook" "kaggle_ingestion" {
  path     = "/Shared/ingest-kaggle-xray-data"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/01_ingestion/ingest_kaggle_xray_data.py"
}

# Outputs

