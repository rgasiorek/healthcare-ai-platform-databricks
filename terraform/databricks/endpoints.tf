# Model Serving Endpoints
# Deploys ML models as REST API endpoints with A/B testing support

# Simple POC Endpoint - Original Model (For Demos)
# This endpoint is used for simple model usage demos
resource "databricks_model_serving" "pneumonia_poc" {
  name = "pneumonia-poc-classifier"

  config {
    # Original POC model (version 1 - simpler, no remote file support)
    served_entities {
      entity_name           = "healthcare_catalog_${var.environment}.models.pneumonia_poc_classifier"
      entity_version        = "1"
      workload_size         = "Small"
      scale_to_zero_enabled = true
    }

    # No environment vars needed (original model doesn't use Files API)
    # No traffic config needed (single model)
    # No inference logging (demo purposes only)
  }

  # Dependencies: Model must be registered in MLflow first
  # This is the original POC model from train_poc_model.py notebook
}

# A/B Testing Endpoint - Champion vs Challenger
resource "databricks_model_serving" "pneumonia_ab_test" {
  name = "pneumonia-classifier-ab-test"

  config {
    # Champion Model (Keras/TensorFlow)
    # Remote file version: accepts Unity Catalog file paths via Files API
    served_entities {
      entity_name           = "healthcare_catalog_${var.environment}.models.pneumonia_poc_classifier_remote_file"
      entity_version        = "9"
      workload_size         = "Small"
      scale_to_zero_enabled = true

      # Environment variables for WorkspaceClient authentication
      environment_vars = {
        "DATABRICKS_HOST"  = var.databricks_workspace_host
        "DATABRICKS_TOKEN" = var.databricks_model_serving_token
      }
    }

    # Challenger Model (PyTorch)
    # Remote file version: accepts Unity Catalog file paths via Files API
    served_entities {
      entity_name           = "healthcare_catalog_${var.environment}.models.pneumonia_poc_classifier_pytorch_remote_file"
      entity_version        = "9"
      workload_size         = "Small"
      scale_to_zero_enabled = true

      # Environment variables for WorkspaceClient authentication
      environment_vars = {
        "DATABRICKS_HOST"  = var.databricks_workspace_host
        "DATABRICKS_TOKEN" = var.databricks_model_serving_token
      }
    }

    # Traffic Split Configuration (A/B Testing)
    traffic_config {
      routes {
        served_model_name  = "pneumonia_poc_classifier_remote_file-9"
        traffic_percentage = 50
      }
      routes {
        served_model_name  = "pneumonia_poc_classifier_pytorch_remote_file-9"
        traffic_percentage = 50
      }
    }

    # Inference Table Logging (CRITICAL for A/B analysis)
    auto_capture_config {
      catalog_name      = "healthcare_catalog_${var.environment}"
      schema_name       = "gold"
      table_name_prefix = "pneumonia_classifier"
      enabled           = true
    }
  }

  # Dependencies: Models must be registered in MLflow first
  # Note: Terraform can't check if MLflow models exist, so this will fail
  # on first apply if models aren't registered yet. Run training notebooks first.
}

# Feedback Endpoint (Production-Ready Feedback Collection)
# DISABLED: Model not deployed yet, feedback handled via Streamlit app
# Uncomment and apply after running deploy_feedback_endpoint.py notebook
#
# resource "databricks_model_serving" "feedback_endpoint" {
#   name = "feedback-endpoint"
#
#   config {
#     served_entities {
#       entity_name    = "healthcare_catalog_${var.environment}.models.feedback_processor"
#       entity_version = "1"
#       workload_size  = "Small"
#       scale_to_zero_enabled = true
#     }
#
#     # No traffic config needed (single model)
#     # No inference logging needed (this endpoint processes feedback, not predictions)
#   }
#
#   lifecycle {
#     ignore_changes = [
#       config[0].served_entities[0].entity_version
#     ]
#   }
#
#   # Dependencies: FeedbackProcessor model must be registered first
#   # Run deploy_feedback_endpoint.py notebook to register model before terraform apply
# }
