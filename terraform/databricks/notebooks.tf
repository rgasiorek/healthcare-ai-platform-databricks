# Databricks Notebooks - Deployed via Terraform
# Notebook source code stored in /notebooks/ directory

# Kaggle X-ray Data Ingestion Notebook
resource "databricks_notebook" "ingest_kaggle_data" {
  path     = "/Shared/ingest-kaggle-xray-data"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/01_ingestion/ingest_kaggle_xray_data.py"
}

# ML POC Training Notebook (TensorFlow/Keras)
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

# Wrap and Register Path-Based Models (Files API - for A/B testing)
resource "databricks_notebook" "wrap_and_register_path_models" {
  path     = "/Shared/wrap_and_register_path_models"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/03_ml/wrap_and_register_path_models.py"
}

# ML Demo: SDK vs REST API Model Usage (Single Model Interaction)
resource "databricks_notebook" "demo_model_usage" {
  path     = "/Shared/demo-model-usage"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/03_ml/demo_model_usage.py"
}

# End-to-End Demo (Full A/B Testing Workflow)
resource "databricks_notebook" "end_to_end_demo" {
  path     = "/Shared/end-to-end-demo"
  language = "PYTHON"
  source   = "${path.module}/../../notebooks/05_demo/end_to_end_demo.py"
}
