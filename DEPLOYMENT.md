# Deployment Guide - Infrastructure as Code

> **Principle**: Notebooks do ML work. Terraform does infrastructure.

---

## Architecture: Proper Separation of Concerns

```
┌─────────────────────────────────────────────────────────────┐
│                    NOTEBOOKS (ML Work)                       │
├─────────────────────────────────────────────────────────────┤
│  1. Train models (Keras, PyTorch)                           │
│  2. Register to MLflow Model Registry                       │
│  3. Test model behavior                                     │
│  4. Log experiments                                         │
│                                                             │
│  DO NOT: Deploy endpoints, create infrastructure           │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ Models registered in MLflow
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  TERRAFORM (Infrastructure)                  │
├─────────────────────────────────────────────────────────────┤
│  1. Reference models by name (assume they exist)            │
│  2. Deploy serving endpoints                                │
│  3. Configure traffic splits (A/B testing)                  │
│  4. Enable inference logging                                │
│  5. Create tables, volumes, clusters                        │
│                                                             │
│  DO NOT: Train models, run ML experiments                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Workflow

### Step 1: Train and Register Models (Notebooks)

Run these notebooks in order:

```bash
# 1. Data ingestion
/Shared/ingest-kaggle-xray-data

# 2. Train Champion model (Keras)
/Shared/train-poc-model

# 3. Train Challenger model (PyTorch)
/Shared/train-poc-model-pytorch

# 4. Register feedback processor (optional)
/Shared/deploy-feedback-endpoint
```

**Result**: Models registered in MLflow Model Registry
- `healthcare_catalog_dev.models.pneumonia_poc_classifier` (v1)
- `healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch` (v1)
- `healthcare_catalog_dev.models.feedback_processor` (v1)

---

### Step 2: Deploy Infrastructure (Terraform)

```bash
cd terraform
terraform plan   # Review changes
terraform apply  # Deploy endpoints
```

**What Terraform Deploys**:

#### 1. A/B Testing Endpoint (`pneumonia-classifier-ab-test`)
```hcl
# terraform/databricks/endpoints.tf
resource "databricks_model_serving" "pneumonia_ab_test" {
  name = "pneumonia-classifier-ab-test"

  config {
    # Champion Model (Keras)
    served_entities {
      entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
      entity_version = "1"
      workload_size  = "Small"
      scale_to_zero_enabled = true
    }

    # Challenger Model (PyTorch)
    served_entities {
      entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
      entity_version = "1"
      workload_size  = "Small"
      scale_to_zero_enabled = true
    }

    # Traffic Split (50/50 A/B testing)
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

    # Inference Logging (auto-capture)
    auto_capture_config {
      catalog_name      = "healthcare_catalog_dev"
      schema_name       = "gold"
      table_name_prefix = "pneumonia_classifier"
      enabled           = true
    }
  }
}
```

#### 2. Feedback Endpoint (`feedback-endpoint`)
```hcl
resource "databricks_model_serving" "feedback_endpoint" {
  name = "feedback-endpoint"

  config {
    served_entities {
      entity_name    = "healthcare_catalog_dev.models.feedback_processor"
      entity_version = "1"
      workload_size  = "Small"
      scale_to_zero_enabled = true
    }
  }
}
```

**Result**:
- ✅ Endpoints deployed and ready in 5-10 minutes
- ✅ Traffic automatically split 50/50
- ✅ Inference logging enabled
- ✅ Infrastructure versioned in Git

---

## Accessing Deployed Endpoints

### Endpoint URLs (from Terraform output):

```bash
terraform output ab_test_endpoint_url
# Output: https://dbc-68a1cdfa-43b8.cloud.databricks.com/serving-endpoints/pneumonia-classifier-ab-test/invocations

terraform output feedback_endpoint_url
# Output: https://dbc-68a1cdfa-43b8.cloud.databricks.com/serving-endpoints/feedback-endpoint/invocations
```

### Making Predictions:

```python
import requests

# Get token from Databricks
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")

# Call A/B testing endpoint
response = requests.post(
    f"https://{workspace_url}/serving-endpoints/pneumonia-classifier-ab-test/invocations",
    headers={"Authorization": f"Bearer {token}"},
    json={"inputs": [image_array.tolist()]}
)

prediction = response.json()['predictions'][0][0]
request_id = response.headers['x-databricks-request-id']
```

---

## Updating Traffic Split

### To Change Traffic Percentages:

Edit `terraform/databricks/endpoints.tf`:

```hcl
traffic_config {
  routes {
    served_model_name   = "pneumonia_poc_classifier-1"
    traffic_percentage  = 90  # ← Promote winner to 90%
  }
  routes {
    served_model_name   = "pneumonia_poc_classifier_pytorch-1"
    traffic_percentage  = 10  # ← Reduce loser to 10%
  }
}
```

Then apply:

```bash
terraform apply
```

**Result**: Zero-downtime traffic shift in ~1-2 minutes

---

## Updating Model Versions

### When You Train a New Model:

1. **Register new model version** (notebook):
   ```python
   # In train-poc-model-pytorch.py
   mlflow.pytorch.log_model(
       model,
       artifact_path="model",
       registered_model_name="healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
       # ↑ This creates version 2
   )
   ```

2. **Update Terraform** to reference new version:
   ```hcl
   served_entities {
     entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
     entity_version = "2"  # ← Changed from "1"
     ...
   }
   ```

3. **Apply**:
   ```bash
   terraform apply
   ```

**Alternative**: Use lifecycle rule to ignore version changes (already in `endpoints.tf`), then update via Databricks UI for quick experiments.

---

## Why This Approach?

### Before (Notebook-based Deployment):
```python
# In notebook: deploy_ab_testing_endpoint.py
response = requests.post(
    "https://.../api/2.0/serving-endpoints",
    json=endpoint_config
)
```

**Problems**:
- ❌ Mixed ML work with infrastructure
- ❌ No version control for infrastructure
- ❌ Manual API calls in notebooks
- ❌ Hard to reproduce deployments
- ❌ Violates IaC principles

### After (Terraform-based Deployment):
```hcl
# In terraform/databricks/endpoints.tf
resource "databricks_model_serving" "pneumonia_ab_test" {
  name = "pneumonia-classifier-ab-test"
  ...
}
```

**Benefits**:
- ✅ Proper separation of concerns
- ✅ Infrastructure versioned in Git
- ✅ Declarative (desired state, not imperative steps)
- ✅ Reproducible across environments
- ✅ Reviewable via pull requests
- ✅ Supports GitOps workflows
- ✅ Follows IaC best practices

---

## Comparison: Old vs New

| Aspect | Old (Notebook) | New (Terraform) |
|--------|---------------|-----------------|
| **Deployment Method** | Notebook API calls | `terraform apply` |
| **Version Control** | None | Git-tracked .tf files |
| **Reproducibility** | Manual re-run | `terraform apply` |
| **Environment Sync** | Manual | Terraform state |
| **Code Review** | No | Pull request review |
| **Rollback** | Manual API calls | `terraform apply` old commit |
| **Documentation** | Inline comments | Self-documenting IaC |
| **Separation of Concerns** | Mixed | Clean (ML vs Infra) |

---

## Troubleshooting

### Error: "Model not found"

**Cause**: Terraform tries to deploy endpoint before models are registered in MLflow.

**Solution**: Run training notebooks first to register models, then run `terraform apply`.

### Error: "Endpoint already exists"

**Cause**: Endpoint was created manually (via notebook or UI) before Terraform.

**Solution**: Import existing endpoint into Terraform state:
```bash
terraform import module.databricks.databricks_model_serving.pneumonia_ab_test pneumonia-classifier-ab-test
```

### Endpoint stuck in "UPDATING" state

**Cause**: Databricks is provisioning containers (takes 5-10 minutes).

**Solution**: Wait and check status:
```bash
terraform refresh
terraform output ab_test_endpoint_url
```

---

## Next Steps

1. ✅ **Models registered** (via notebooks)
2. ✅ **Endpoints deployed** (via Terraform)
3. ⏭️ **Make predictions** (via REST API or notebook)
4. ⏭️ **Submit feedback** (via feedback endpoint)
5. ⏭️ **Monitor performance** (via SQL queries / BI dashboard)
6. ⏭️ **Promote winner** (update traffic split in Terraform)

Follow **EXECUTION_GUIDE.md** for complete end-to-end workflow.
