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

# ML POC Training Notebook (TensorFlow)
resource "databricks_notebook" "train_poc_model" {
  path     = "/Shared/train-poc-model"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/03_ml/train_poc_model.py"
}

# ML POC Training Notebook (PyTorch)
resource "databricks_notebook" "train_poc_model_pytorch" {
  path     = "/Shared/train-poc-model-pytorch"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/03_ml/train_poc_model_pytorch.py"
}

# ML POC Serving Endpoint Notebook
resource "databricks_notebook" "deploy_serving_endpoint" {
  path     = "/Shared/deploy-serving-endpoint"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/03_ml/deploy_serving_endpoint.py"
}

# ML Demo: SDK vs REST API Model Usage
resource "databricks_notebook" "demo_model_usage" {
  path     = "/Shared/demo-model-usage"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/03_ml/demo_model_usage.py"
}

# Outputs

