# Databricks Jobs - Automated Workflows
# Jobs provide production-ready automation for notebook execution

# Data source: Current user (for job ownership)
data "databricks_current_user" "me" {}

# Job: Deploy Model Serving Endpoint
# This job automates the deployment of the pneumonia classifier model to a serving endpoint
resource "databricks_job" "deploy_serving_endpoint" {
  name = "Deploy Pneumonia Serving Endpoint - ${var.environment}"

  # Job description
  description = <<-EOT
    Automated deployment of pneumonia classification model to serving endpoint.

    This job:
    1. Checks if serving endpoint exists
    2. Creates endpoint if missing (or updates if exists)
    3. Waits for endpoint to be READY
    4. Tests endpoint with sample predictions

    **Educational Note:**
    - The notebook can be run manually for learning/debugging
    - This job automates the process for production deployments
    - Can be triggered via API, schedule, or manually from UI
  EOT

  # Task: Run the deployment notebook
  task {
    task_key = "deploy_endpoint"

    # Use the existing deployment notebook
    notebook_task {
      notebook_path = databricks_notebook.deploy_serving_endpoint.path
      source        = "WORKSPACE"
    }

    # Job cluster configuration (cost-effective, spins up when needed)
    new_cluster {
      spark_version      = "15.4.x-scala2.12"  # Latest LTS
      node_type_id       = "i3.xlarge"
      num_workers        = 0  # Single-node (no workers needed for API calls)

      # Enable Unity Catalog
      data_security_mode = "SINGLE_USER"

      spark_conf = {
        "spark.databricks.delta.preview.enabled" = "true"
      }

      custom_tags = {
        "Project"     = "healthcare-xray"
        "Environment" = var.environment
        "JobType"     = "ml-deployment"
      }
    }

    # Retry configuration
    max_retries          = 1
    timeout_seconds      = 3600  # 1 hour max (endpoint deployment can take 5-10 min)
    min_retry_interval_millis = 60000  # Wait 1 min before retry
  }

  # Email notifications (optional - configure with actual email)
  # email_notifications {
  #   on_failure = ["your-email@example.com"]
  #   on_success = ["your-email@example.com"]
  # }

  # Job parameters (can be overridden when triggering)
  parameter {
    name    = "model_name"
    default = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
  }

  parameter {
    name    = "model_version"
    default = "1"
  }

  parameter {
    name    = "endpoint_name"
    default = "pneumonia-poc-classifier"
  }

  # Tags for organization
  tags = {
    type        = "deployment"
    model       = "pneumonia-classifier"
    environment = var.environment
  }

  depends_on = [
    databricks_notebook.deploy_serving_endpoint,
    databricks_catalog.healthcare,
    databricks_schema.models
  ]
}

# Job: Deploy A/B Testing Endpoint (Champion/Challenger)
# This job automates deployment of both models with traffic splitting
resource "databricks_job" "deploy_ab_testing_endpoint" {
  name = "Deploy A/B Testing Endpoint - ${var.environment}"

  description = <<-EOT
    Automated deployment of Champion/Challenger A/B testing endpoint.

    This job:
    1. Verifies both models exist (Keras Champion + PyTorch Challenger)
    2. Creates/updates endpoint with traffic splitting (50/50 initially)
    3. Enables inference table logging (critical for performance analysis)
    4. Tests traffic distribution
    5. Verifies both models receiving requests

    **Champion/Challenger Pattern:**
    - Champion: Current production model (proven, majority traffic)
    - Challenger: New model being tested (experimental, minority traffic)
    - Monitor performance → Promote winner → Continuous improvement

    **Educational Note:**
    - Shows real-world MLOps A/B testing
    - Enables data-driven model promotion decisions
    - Demonstrates gradual rollout strategy
  EOT

  task {
    task_key = "deploy_ab_endpoint"

    notebook_task {
      notebook_path = databricks_notebook.deploy_ab_testing_endpoint.path
      source        = "WORKSPACE"
    }

    new_cluster {
      spark_version      = "15.4.x-scala2.12"
      node_type_id       = "i3.xlarge"
      num_workers        = 0  # Single-node

      data_security_mode = "SINGLE_USER"

      spark_conf = {
        "spark.databricks.delta.preview.enabled" = "true"
      }

      custom_tags = {
        "Project"     = "healthcare-xray"
        "Environment" = var.environment
        "JobType"     = "ml-ab-testing"
      }
    }

    max_retries               = 1
    timeout_seconds           = 3600
    min_retry_interval_millis = 60000
  }

  parameter {
    name    = "champion_model"
    default = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
  }

  parameter {
    name    = "challenger_model"
    default = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
  }

  parameter {
    name    = "champion_traffic_pct"
    default = "50"
  }

  parameter {
    name    = "challenger_traffic_pct"
    default = "50"
  }

  tags = {
    type        = "ab-testing"
    pattern     = "champion-challenger"
    environment = var.environment
  }

  depends_on = [
    databricks_notebook.deploy_ab_testing_endpoint,
    databricks_catalog.healthcare,
    databricks_schema.models,
    databricks_schema.gold
  ]
}

# Output job information
output "deploy_job_id" {
  description = "ID of the model deployment job (single model)"
  value       = databricks_job.deploy_serving_endpoint.id
}

output "deploy_job_url" {
  description = "URL to view the deployment job in Databricks workspace"
  value       = databricks_job.deploy_serving_endpoint.url
}

output "ab_testing_job_id" {
  description = "ID of the A/B testing deployment job (Champion/Challenger)"
  value       = databricks_job.deploy_ab_testing_endpoint.id
}

output "ab_testing_job_url" {
  description = "URL to view the A/B testing job in Databricks workspace"
  value       = databricks_job.deploy_ab_testing_endpoint.url
}
