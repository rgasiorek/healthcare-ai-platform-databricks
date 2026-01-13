# Databricks notebook source
# MAGIC %md
# MAGIC # Champion/Challenger A/B Testing Endpoint - DEPRECATED
# MAGIC
# MAGIC ⚠️ **This notebook is DEPRECATED for endpoint deployment.**
# MAGIC
# MAGIC ## New Architecture (Infrastructure as Code)
# MAGIC
# MAGIC **Endpoint deployment is now handled by Terraform** (`terraform/databricks/endpoints.tf`).
# MAGIC
# MAGIC This follows proper separation of concerns:
# MAGIC - **Notebooks** (ML work): Train models, register to MLflow
# MAGIC - **Terraform** (Infrastructure): Deploy endpoints, configure traffic
# MAGIC
# MAGIC ### To Deploy Endpoints:
# MAGIC
# MAGIC ```bash
# MAGIC # After training models (run training notebooks first):
# MAGIC terraform apply
# MAGIC ```
# MAGIC
# MAGIC Terraform will automatically:
# MAGIC 1. Deploy A/B testing endpoint (`pneumonia-classifier-ab-test`)
# MAGIC 2. Configure 50/50 traffic split
# MAGIC 3. Enable inference logging
# MAGIC 4. Deploy feedback endpoint (`feedback-endpoint`)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## What This Notebook Still Does
# MAGIC
# MAGIC This notebook is now **read-only documentation** showing:
# MAGIC - How A/B testing works
# MAGIC - How to verify endpoint status
# MAGIC - How to test traffic distribution
# MAGIC - Educational content about Champion/Challenger pattern
# MAGIC
# MAGIC **Champion/Challenger Pattern**:
# MAGIC - **Champion**: Current production model (e.g., Keras) - gets 50% traffic
# MAGIC - **Challenger**: New model being tested (e.g., PyTorch) - gets 50% traffic
# MAGIC - Monitor performance → Promote winner → Gradually shift traffic
# MAGIC
# MAGIC **What This Enables**:
# MAGIC 1. Compare TensorFlow vs PyTorch in production
# MAGIC 2. Measure which framework performs better
# MAGIC 3. Make data-driven promotion decisions
# MAGIC 4. Zero-downtime model updates
# MAGIC
# MAGIC **Prerequisites**:
# MAGIC - Both models registered in MLflow Model Registry (train notebooks)
# MAGIC - Endpoint deployed via Terraform (not this notebook!)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Configuration and Imports

# COMMAND ----------
import requests
import json
import time
from datetime import datetime

# Configuration
ENDPOINT_NAME = "pneumonia-classifier-ab-test"

# Champion Model (Current Production)
CHAMPION_MODEL_NAME = "healthcare_catalog_dev.models.pneumonia_poc_classifier"  # Keras
CHAMPION_MODEL_VERSION = "1"
CHAMPION_TRAFFIC_PCT = 50  # 50% traffic initially

# Challenger Model (Testing)
CHALLENGER_MODEL_NAME = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"  # PyTorch
CHALLENGER_MODEL_VERSION = "1"
CHALLENGER_TRAFFIC_PCT = 50  # 50% traffic initially

# Inference Logging Configuration
INFERENCE_LOG_CATALOG = "healthcare_catalog_dev"
INFERENCE_LOG_SCHEMA = "gold"
INFERENCE_LOG_TABLE_PREFIX = "pneumonia_classifier"  # Will create *_payload and *_predictions tables

# Workload size
WORKLOAD_SIZE = "Small"  # Cost-effective for POC

# Get workspace details
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

print(f"Configuration:")
print(f"  Workspace URL: {workspace_url}")
print(f"  Endpoint Name: {ENDPOINT_NAME}")
print(f"\n  CHAMPION:")
print(f"    Model: {CHAMPION_MODEL_NAME}")
print(f"    Version: {CHAMPION_MODEL_VERSION}")
print(f"    Traffic: {CHAMPION_TRAFFIC_PCT}%")
print(f"\n  CHALLENGER:")
print(f"    Model: {CHALLENGER_MODEL_NAME}")
print(f"    Version: {CHALLENGER_MODEL_VERSION}")
print(f"    Traffic: {CHALLENGER_TRAFFIC_PCT}%")
print(f"\n  Inference Logging:")
print(f"    Tables: {INFERENCE_LOG_CATALOG}.{INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE_PREFIX}_*")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Verify Models Exist in MLflow Registry

# COMMAND ----------
import mlflow
from mlflow import MlflowClient

mlflow.set_tracking_uri("databricks")
client = MlflowClient()

print("Verifying models exist in MLflow Registry...\n")

# Check Champion model
try:
    champion_model = client.get_registered_model(CHAMPION_MODEL_NAME)
    champion_versions = client.search_model_versions(f"name='{CHAMPION_MODEL_NAME}'")
    print(f"✅ Champion Model: {CHAMPION_MODEL_NAME}")
    print(f"   Versions: {[v.version for v in champion_versions]}")
except Exception as e:
    print(f"❌ Champion model not found: {e}")
    dbutils.notebook.exit(f"ERROR: Champion model {CHAMPION_MODEL_NAME} not found")

# Check Challenger model
try:
    challenger_model = client.get_registered_model(CHALLENGER_MODEL_NAME)
    challenger_versions = client.search_model_versions(f"name='{CHALLENGER_MODEL_NAME}'")
    print(f"\n✅ Challenger Model: {CHALLENGER_MODEL_NAME}")
    print(f"   Versions: {[v.version for v in challenger_versions]}")
except Exception as e:
    print(f"❌ Challenger model not found: {e}")
    dbutils.notebook.exit(f"ERROR: Challenger model {CHALLENGER_MODEL_NAME} not found")

print("\n✅ Both models exist and ready for deployment!")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: ~~Create or Update A/B Testing Endpoint~~ DEPRECATED
# MAGIC
# MAGIC **This step is now handled by Terraform!**
# MAGIC
# MAGIC ### Old Approach (Deprecated):
# MAGIC - Notebook deployed endpoint via Databricks REST API
# MAGIC - Mixed ML work (notebooks) with infrastructure (endpoints)
# MAGIC - Violates infrastructure-as-code principles
# MAGIC
# MAGIC ### New Approach (Current):
# MAGIC ```bash
# MAGIC # Terraform manages endpoint deployment
# MAGIC terraform apply
# MAGIC ```
# MAGIC
# MAGIC ### Terraform Configuration (terraform/databricks/endpoints.tf):
# MAGIC ```hcl
# MAGIC resource "databricks_model_serving" "pneumonia_ab_test" {
# MAGIC   name = "pneumonia-classifier-ab-test"
# MAGIC
# MAGIC   config {
# MAGIC     served_entities {
# MAGIC       entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
# MAGIC       entity_version = "1"
# MAGIC       workload_size  = "Small"
# MAGIC       scale_to_zero_enabled = true
# MAGIC     }
# MAGIC
# MAGIC     served_entities {
# MAGIC       entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
# MAGIC       entity_version = "1"
# MAGIC       workload_size  = "Small"
# MAGIC       scale_to_zero_enabled = true
# MAGIC     }
# MAGIC
# MAGIC     traffic_config {
# MAGIC       routes {
# MAGIC         served_model_name   = "pneumonia_poc_classifier-1"
# MAGIC         traffic_percentage  = 50
# MAGIC       }
# MAGIC       routes {
# MAGIC         served_model_name   = "pneumonia_poc_classifier_pytorch-1"
# MAGIC         traffic_percentage  = 50
# MAGIC       }
# MAGIC     }
# MAGIC
# MAGIC     auto_capture_config {
# MAGIC       catalog_name      = "healthcare_catalog_dev"
# MAGIC       schema_name       = "gold"
# MAGIC       table_name_prefix = "pneumonia_classifier"
# MAGIC       enabled           = true
# MAGIC     }
# MAGIC   }
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Benefits**:
# MAGIC - ✅ Infrastructure as code (versioned, reviewable)
# MAGIC - ✅ Separation of concerns (ML vs Infrastructure)
# MAGIC - ✅ Reproducible deployments
# MAGIC - ✅ No manual API calls in notebooks

# COMMAND ----------
print("⚠️  STEP 3 DEPRECATED")
print("")
print("Endpoint deployment is now managed by Terraform.")
print("Run 'terraform apply' to deploy the A/B testing endpoint.")
print("")
print("This notebook now focuses on:")
print("  - Verifying endpoint status (Step 4)")
print("  - Testing traffic distribution (Step 5)")
print("  - Educational content about A/B testing")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Monitor Endpoint Deployment Status

# COMMAND ----------
# Check endpoint status
endpoint_url = f"https://{workspace_url}/api/2.0/serving-endpoints/{ENDPOINT_NAME}"

response = requests.get(
    endpoint_url,
    headers={"Authorization": f"Bearer {token}"}
)

if response.status_code == 200:
    endpoint_info = response.json()
    status = endpoint_info.get("state", {}).get("ready", "UNKNOWN")
    config_update = endpoint_info.get("state", {}).get("config_update", "UNKNOWN")

    print(f"Endpoint Status:")
    print(f"  Name: {ENDPOINT_NAME}")
    print(f"  Ready: {status}")
    print(f"  Config Update: {config_update}")

    # Show served models
    served_entities = endpoint_info.get("config", {}).get("served_entities", [])
    print(f"\n  Served Models ({len(served_entities)}):")
    for entity in served_entities:
        print(f"    - {entity.get('entity_name')} (v{entity.get('entity_version')})")

    # Show traffic config
    traffic_config = endpoint_info.get("config", {}).get("traffic_config", {})
    routes = traffic_config.get("routes", [])
    print(f"\n  Traffic Distribution:")
    for route in routes:
        print(f"    - {route.get('served_model_name')}: {route.get('traffic_percentage')}%")

    # Show inference logging config
    auto_capture = endpoint_info.get("config", {}).get("auto_capture_config", {})
    if auto_capture.get("enabled"):
        print(f"\n  ✅ Inference Logging Enabled:")
        print(f"    Tables: {auto_capture.get('catalog_name')}.{auto_capture.get('schema_name')}.{auto_capture.get('table_name_prefix')}_*")
    else:
        print(f"\n  ⚠️  Inference logging NOT enabled")

    if status == "READY":
        print(f"\n✅ Endpoint is READY! Proceed to Step 5 to test A/B traffic split.")
    else:
        print(f"\n⏳ Endpoint not ready yet. Wait a few minutes and re-run this cell.")
        print(f"\nTip: You can also check status in Databricks UI:")
        print(f"  Navigate to: Serving → {ENDPOINT_NAME}")
else:
    print(f"❌ Error fetching endpoint status: {response.status_code}")
    print(f"Response: {response.text}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Test A/B Traffic Split
# MAGIC
# MAGIC **Run this ONLY after endpoint is READY (Step 4 shows green status)**
# MAGIC
# MAGIC We'll make 10 predictions and verify both models receive traffic

# COMMAND ----------
from PIL import Image
import numpy as np

# Load a test image
sample_df = spark.sql("""
    SELECT * FROM healthcare_catalog_dev.bronze.kaggle_xray_metadata
    WHERE category = 'NORMAL' LIMIT 1
""")

test_row = sample_df.first()
test_image_path = test_row.file_path
true_category = test_row.category

# Load and preprocess image
local_path = test_image_path.replace("dbfs:", "")
img = Image.open(local_path)
img = img.convert('RGB')
img = img.resize((64, 64))
img_array = np.array(img) / 255.0

print(f"Test Image:")
print(f"  Path: {test_image_path}")
print(f"  True Category: {true_category}")
print(f"  Image shape: {img_array.shape}")

# COMMAND ----------
# Make 10 predictions to see traffic distribution
invocation_url = f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"

payload = {
    "inputs": [img_array.tolist()]
}

print(f"Making 10 predictions to test A/B traffic split...\n")
print(f"{'='*80}")

model_counts = {}
prediction_ids = []

for i in range(10):
    try:
        response = requests.post(
            invocation_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()

            # Extract prediction
            prediction = result.get('predictions', [[]])[0]
            pred_prob = prediction[0] if isinstance(prediction, list) else prediction
            pred_label = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"

            # Get request ID (for feedback linking)
            request_id = response.headers.get("x-databricks-request-id", "unknown")
            prediction_ids.append(request_id)

            # Note: served_model_name might not be in response, but it's logged in inference tables
            # We'll verify distribution by checking inference tables later

            print(f"Prediction {i+1}: {pred_label} (prob={pred_prob:.3f}) | ID: {request_id[:20]}...")

        else:
            print(f"Prediction {i+1}: ❌ Error {response.status_code}")

    except Exception as e:
        print(f"Prediction {i+1}: ❌ Exception: {e}")

    time.sleep(0.5)  # Small delay between requests

print(f"{'='*80}")
print(f"\n✅ Test complete! Made 10 predictions.")
print(f"\n💡 To see which model served each request, check inference tables:")
print(f"   SELECT request_id, served_model_name FROM {INFERENCE_LOG_CATALOG}.{INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE_PREFIX}_predictions")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Verify Traffic Distribution in Inference Tables

# COMMAND ----------
# Wait a moment for inference logs to be written
print("Waiting 10 seconds for inference logs to be written...")
time.sleep(10)

# Query inference table to see traffic distribution
try:
    traffic_dist = spark.sql(f"""
        SELECT
            served_model_name,
            COUNT(*) as request_count,
            ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as traffic_pct
        FROM {INFERENCE_LOG_CATALOG}.{INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE_PREFIX}_predictions
        WHERE date >= current_date()
        GROUP BY served_model_name
        ORDER BY request_count DESC
    """)

    print(f"\n{'='*80}")
    print(f"TRAFFIC DISTRIBUTION (Today)")
    print(f"{'='*80}\n")

    traffic_dist.show(truncate=False)

    print(f"\n✅ Verify both models are receiving traffic!")
    print(f"   Expected: ~{CHAMPION_TRAFFIC_PCT}% Champion, ~{CHALLENGER_TRAFFIC_PCT}% Challenger")
    print(f"   (Distribution may vary with small sample size)")

except Exception as e:
    print(f"⚠️  Could not query inference tables (they may not exist yet): {e}")
    print(f"\nInference tables are created automatically by Databricks.")
    print(f"If this is the first deployment, wait a few minutes and try again.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Show Sample Predictions with Model Attribution

# COMMAND ----------
# Show recent predictions with which model served them
try:
    recent_predictions = spark.sql(f"""
        SELECT
            request_id,
            served_model_name,
            timestamp_ms,
            CAST(response:predictions[0][0] AS DOUBLE) as prediction_score,
            CASE
                WHEN CAST(response:predictions[0][0] AS DOUBLE) > 0.5 THEN 'PNEUMONIA'
                ELSE 'NORMAL'
            END as predicted_class
        FROM {INFERENCE_LOG_CATALOG}.{INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE_PREFIX}_predictions
        WHERE date >= current_date()
        ORDER BY timestamp_ms DESC
        LIMIT 10
    """)

    print(f"\n{'='*80}")
    print(f"RECENT PREDICTIONS (Last 10)")
    print(f"{'='*80}\n")

    recent_predictions.show(truncate=False)

    print(f"\n💡 Key Insight:")
    print(f"   The 'served_model_name' column shows which model handled each request")
    print(f"   This is CRITICAL for comparing Champion vs Challenger performance!")

except Exception as e:
    print(f"Could not query recent predictions: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary: A/B Testing Endpoint Deployed! ✅
# MAGIC
# MAGIC ### What We Accomplished:
# MAGIC 1. ✅ Deployed single endpoint serving TWO models (Champion + Challenger)
# MAGIC 2. ✅ Configured traffic split (50/50 initially, adjustable)
# MAGIC 3. ✅ Enabled inference table logging (captures which model served each request)
# MAGIC 4. ✅ Tested traffic distribution
# MAGIC 5. ✅ Verified both models receiving requests
# MAGIC
# MAGIC ### Endpoint Details:
# MAGIC - **Name**: `pneumonia-classifier-ab-test`
# MAGIC - **URL**: `https://{workspace_url}/serving-endpoints/pneumonia-classifier-ab-test/invocations`
# MAGIC - **Champion**: {CHAMPION_MODEL_NAME} ({CHAMPION_TRAFFIC_PCT}% traffic)
# MAGIC - **Challenger**: {CHALLENGER_MODEL_NAME} ({CHALLENGER_TRAFFIC_PCT}% traffic)
# MAGIC - **Inference Logs**: `{INFERENCE_LOG_CATALOG}.{INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE_PREFIX}_*`
# MAGIC
# MAGIC ### Inference Tables Created:
# MAGIC 1. `{INFERENCE_LOG_CATALOG}.{INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE_PREFIX}_payload` - Request inputs
# MAGIC 2. `{INFERENCE_LOG_CATALOG}.{INFERENCE_LOG_SCHEMA}.{INFERENCE_LOG_TABLE_PREFIX}_predictions` - Responses + **served_model_name**
# MAGIC
# MAGIC ### Next Steps (Complete A/B Testing Workflow):
# MAGIC
# MAGIC 1. **Collect Feedback** (Issue #13):
# MAGIC    - Radiologists review predictions
# MAGIC    - Use feedback_collector to submit ground truth
# MAGIC    - Links to `request_id` from inference table
# MAGIC
# MAGIC 2. **Monitor Performance** (Issue #14):
# MAGIC    - Query `model_performance_live` view
# MAGIC    - Compare Champion vs Challenger accuracy
# MAGIC    - Calculate statistical significance
# MAGIC
# MAGIC 3. **Make Decision**:
# MAGIC    - If Challenger wins: Increase traffic (90% Challenger, 10% Champion)
# MAGIC    - If tied: Keep testing with 50/50
# MAGIC    - If Champion wins: Reduce Challenger traffic or retire
# MAGIC
# MAGIC 4. **Promote Winner**:
# MAGIC    ```python
# MAGIC    # Update traffic to 100% best model
# MAGIC    # Champion = winner, Challenger = new experiment
# MAGIC    ```
# MAGIC
# MAGIC ### How to Adjust Traffic Split:
# MAGIC
# MAGIC Simply update the percentages in this notebook and re-run:
# MAGIC ```python
# MAGIC CHAMPION_TRAFFIC_PCT = 90   # Give 90% to current best
# MAGIC CHALLENGER_TRAFFIC_PCT = 10  # Give 10% to test new model
# MAGIC ```
# MAGIC
# MAGIC ### Educational Notes for Pupils:
# MAGIC
# MAGIC **Why A/B Testing?**
# MAGIC - Compare models in production with real data
# MAGIC - Reduce risk (gradual rollout, not all-or-nothing)
# MAGIC - Make data-driven decisions (not gut feelings)
# MAGIC - Learn which approach works better (TensorFlow vs PyTorch)
# MAGIC
# MAGIC **Champion/Challenger Pattern:**
# MAGIC - Champion: Proven, safe, gets majority traffic
# MAGIC - Challenger: New, experimental, gets minority traffic
# MAGIC - If Challenger wins → becomes new Champion
# MAGIC - Continuous improvement cycle
# MAGIC
# MAGIC **Key Metric**: Which model has higher accuracy when feedback is collected?
# MAGIC
# MAGIC **POC Complete! Now ready for real-world A/B testing with feedback loop! 🚀**
