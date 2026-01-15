# Model Serving Endpoints
# Deploys ML models as REST API endpoints with A/B testing support

# A/B Testing Endpoint - Champion vs Challenger
resource "databricks_model_serving" "pneumonia_ab_test" {
  name = "pneumonia-classifier-ab-test"

  config {
    # Champion Model (Keras/TensorFlow)
    served_entities {
      entity_name    = "healthcare_catalog_${var.environment}.models.pneumonia_poc_classifier"
      entity_version = "1"
      workload_size  = "Small"
      scale_to_zero_enabled = true
    }

    # Challenger Model (PyTorch)
    served_entities {
      entity_name    = "healthcare_catalog_${var.environment}.models.pneumonia_poc_classifier_pytorch"
      entity_version = "1"
      workload_size  = "Small"
      scale_to_zero_enabled = true
    }

    # Traffic Split Configuration (A/B Testing)
    traffic_config {
      routes {
        served_model_name   = "pneumonia_poc_classifier-1"
        traffic_percentage  = 50
      }
      routes {
        served_model_name   = "pneumonia_poc_classifier_pytorch-1"
        traffic_percentage  = 50
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

  # Lifecycle: Ignore entity version changes (allow manual model updates via UI/API)
  lifecycle {
    ignore_changes = [
      config[0].served_entities[0].entity_version,
      config[0].served_entities[1].entity_version
    ]
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
