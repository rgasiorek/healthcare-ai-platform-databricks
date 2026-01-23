# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End MLOps Demo (30 Minutes)
# MAGIC
# MAGIC Complete workflow using Terraform-defined tables:
# MAGIC 1. Warm up endpoint (cold start: 2-3 min)
# MAGIC 2. Make predictions via A/B endpoint
# MAGIC 3. Submit radiologist feedback
# MAGIC 4. Monitor Champion vs Challenger performance
# MAGIC
# MAGIC **Tables Used** (from Terraform):
# MAGIC - `gold.pneumonia_predictions` (predictions from models)
# MAGIC - `gold.prediction_feedback` (ground truth from radiologists)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Prerequisites
# MAGIC
# MAGIC Before running this demo, ensure the following steps are completed:
# MAGIC
# MAGIC 1. **Data Ingested** - X-ray images uploaded to Unity Catalog
# MAGIC    - Run: `/Shared/ingest-kaggle-xray-data` notebook
# MAGIC    - Location: `notebooks/01_ingestion/ingest_kaggle_xray_data.py`
# MAGIC
# MAGIC 2. **Models Trained** - Both Keras and PyTorch models trained and registered
# MAGIC    - Run: `/Shared/train-poc-model` notebook (Keras)
# MAGIC    - Run: `/Shared/train-poc-model-pytorch` notebook (PyTorch)
# MAGIC    - Location: `notebooks/03_ml/train_poc_model.py` and `train_poc_model_pytorch.py`
# MAGIC
# MAGIC 3. **Models Wrapped** - Inference tracking wrapper added to models
# MAGIC    - Run: `/Shared/wrap_and_register_path_models` notebook
# MAGIC    - Location: `notebooks/03_ml/wrap_and_register_path_models.py`
# MAGIC
# MAGIC 4. **Model Serving Deployed** - A/B test endpoint deployed via Terraform
# MAGIC    - Endpoint: `pneumonia-classifier-ab-test`
# MAGIC    - Run: `cd terraform && terraform apply`
# MAGIC    - Reference: `terraform/databricks/endpoints.tf`
# MAGIC
# MAGIC 5. **Feedback App Running** - Radiologist feedback review app deployed
# MAGIC    - App: `radiologist-feedback-review`
# MAGIC    - Deploy: `databricks apps deploy radiologist-feedback-review`
# MAGIC    - Reference: `README.md` in `apps/feedback_review/` directory

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import requests
import json
import uuid
from datetime import datetime
from pyspark.sql.types import *
from pyspark.sql.functions import *

# Configuration
CATALOG = "healthcare_catalog_dev"
SCHEMA_GOLD = "gold"
SCHEMA_BRONZE = "bronze"

# Terraform-defined tables
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA_GOLD}.pneumonia_predictions"
FEEDBACK_TABLE = f"{CATALOG}.{SCHEMA_GOLD}.prediction_feedback"

# A/B Testing Endpoint (deployed via Terraform)
ENDPOINT_NAME = "pneumonia-classifier-ab-test"

# Get workspace credentials
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
invocation_url = f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"

print("Configuration:")
print(f"  Endpoint: {ENDPOINT_NAME}")
print(f"  Predictions Table: {PREDICTIONS_TABLE}")
print(f"  Feedback Table: {FEEDBACK_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Warm Up Endpoint (First Request)
# MAGIC
# MAGIC **Why warm-up is needed:**
# MAGIC - Endpoint configured with `scale_to_zero_enabled = true` in Terraform
# MAGIC - **Cost savings**: Endpoint automatically shuts down when idle (no compute charges)
# MAGIC - **Security**: Inactive endpoints are not running (reduced attack surface)
# MAGIC - **Trade-off**: First request takes 2-3 minutes to start up (cold start)
# MAGIC
# MAGIC Once warm, subsequent requests complete in seconds.

# COMMAND ----------

print("Warming up endpoint (this may take 2-3 minutes)...")
print("  Endpoint configured with scale_to_zero_enabled = true")
print("  - Cost savings: no charges when idle")
print("  - Security: endpoint shut down when not in use")
print("  Waiting for cold start...")

# Send a warmup request with a real file path
# Use first image from bronze as warmup
warmup_image = spark.sql(f"""
    SELECT file_path FROM {CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata LIMIT 1
""").collect()[0].file_path
warmup_payload = {"dataframe_records": [{"file_path": warmup_image}]}

try:
    warmup_response = requests.post(
        invocation_url,
        headers={"Authorization": f"Bearer {token}"},
        json=warmup_payload,
        timeout=300  # 5 minutes for cold start
    )
    if warmup_response.status_code == 200:
        print("Endpoint is warm and ready!")
    else:
        print(f"Warm-up returned status {warmup_response.status_code}")
        print(f"  Response: {warmup_response.text}")
except Exception as e:
    print(f"Warm-up failed: {e}")
    print("  The endpoint may still be starting up. Try running the next cells in a few minutes.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Make Predictions (5-10 predictions)
# MAGIC
# MAGIC **Path-Based Approach**: Send file paths to endpoint (not image bytes)
# MAGIC - More efficient: no network overhead sending image data
# MAGIC - Models handle preprocessing: no type conversion issues
# MAGIC - Better separation: each model controls its own preprocessing

# COMMAND ----------

# Load test images from bronze
test_images = spark.sql(f"""
    SELECT image_id, filename, category, file_path
    FROM {CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata
    LIMIT 10
""").collect()

print(f"Loaded {len(test_images)} test images")
print("-" * 80)

# COMMAND ----------

# Make predictions and write to gold.pneumonia_predictions
# NEW APPROACH: Send file paths instead of image bytes
# - More efficient (no network overhead)
# - Models handle their own preprocessing
# - No type conversion issues
predictions = []

# First request may take up to 3 minutes for cold start
first_request = True

for img in test_images:
    # Call A/B endpoint with FILE PATH (not image bytes!)
    payload = {"dataframe_records": [{"file_path": img.file_path}]}

    # Use longer timeout for first request (cold start), shorter for subsequent
    timeout = 180 if first_request else 60

    try:
        response = requests.post(
            invocation_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=timeout
        )

        first_request = False  # After first successful request, reduce timeout

        if response.status_code == 200:
            result = response.json()
            pred_prob = float(result['predictions'][0][0])  # Ensure it's a Python float
            predicted_label = 1 if pred_prob > 0.5 else 0

            # Use databricks-request-id from response headers (matches inference logs table)
            prediction_id = response.headers.get('databricks-request-id', f"pred-{uuid.uuid4().hex[:12]}")

            # Calculate confidence (distance from 0.5 threshold)
            confidence_score = pred_prob if pred_prob > 0.5 else (1 - pred_prob)

            # Prepare record for our Terraform table
            prediction_record = {
                'prediction_id': prediction_id,
                'image_id': img.image_id,
                'predicted_label': predicted_label,
                'prediction_probability': pred_prob,
                'confidence_score': confidence_score,
                'true_label': 1 if img.category == 'PNEUMONIA' else 0,  # We know true label from bronze
                'is_correct': (predicted_label == (1 if img.category == 'PNEUMONIA' else 0)),
                'model_name': ENDPOINT_NAME,  # A/B endpoint name
                'model_version': '7',  # Both Keras and PyTorch models in A/B test are v7 (Files API)
                'predicted_at': datetime.now(),
                'prediction_date': datetime.now().date()
            }

            predictions.append(prediction_record)

            match = "[MATCH]" if prediction_record['is_correct'] else "[DIFF]"
            print(f"{match} {img.filename}")
            print(f"  True: {img.category} | Predicted: {'PNEUMONIA' if predicted_label == 1 else 'NORMAL'}")
            print(f"  Probability: {pred_prob:.3f} | ID: {prediction_id}")
            print()

    except Exception as e:
        print(f"Error predicting {img.filename}: {e}")

print(f"\nMade {len(predictions)} predictions")

# COMMAND ----------

# Write predictions to gold.pneumonia_predictions (Terraform table)
if predictions:
    # Define schema to match Terraform table exactly
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, TimestampType, DateType

    schema = StructType([
        StructField("prediction_id", StringType(), True),
        StructField("image_id", StringType(), True),
        StructField("predicted_label", IntegerType(), True),
        StructField("prediction_probability", DoubleType(), True),
        StructField("confidence_score", DoubleType(), True),
        StructField("true_label", IntegerType(), True),
        StructField("is_correct", BooleanType(), True),
        StructField("model_name", StringType(), True),
        StructField("model_version", StringType(), True),
        StructField("predicted_at", TimestampType(), True),
        StructField("prediction_date", DateType(), True)
    ])

    pred_df = spark.createDataFrame(predictions, schema=schema)

    pred_df.write.mode('append').saveAsTable(PREDICTIONS_TABLE)

    print(f"Wrote {len(predictions)} predictions to {PREDICTIONS_TABLE}")
else:
    print("No predictions to write")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Verify Predictions in Table

# COMMAND ----------

# Query our Terraform table
recent_predictions = spark.sql(f"""
    SELECT
        prediction_id,
        image_id,
        CASE WHEN predicted_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as predicted,
        CASE WHEN true_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as true_diagnosis,
        ROUND(prediction_probability, 3) as probability,
        is_correct,
        model_name,
        predicted_at
    FROM {PREDICTIONS_TABLE}
    ORDER BY predicted_at DESC
    LIMIT 20
""")

print(f"\nRecent Predictions from {PREDICTIONS_TABLE}:")
print("=" * 120)
recent_predictions.show(truncate=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Submit Radiologist Feedback
# MAGIC
# MAGIC **Use the Streamlit App for interactive feedback submission:**
# MAGIC
# MAGIC ```bash
# MAGIC cd /path/to/apps/feedback_review
# MAGIC streamlit run app.py
# MAGIC ```
# MAGIC
# MAGIC The Streamlit app provides:
# MAGIC - Editable table interface
# MAGIC - Dropdown selectors for diagnosis and confidence
# MAGIC - Real-time validation
# MAGIC - Auto-save to database
# MAGIC
# MAGIC **Or use the automated batch feedback below for quick testing:**

# COMMAND ----------

# Get predictions that don't have feedback yet
pending_feedback = spark.sql(f"""
    SELECT
        p.prediction_id,
        p.image_id,
        CASE WHEN p.predicted_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as ai_prediction,
        CASE WHEN p.true_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as actual_diagnosis,
        p.prediction_probability,
        p.predicted_at
    FROM {PREDICTIONS_TABLE} p
    LEFT JOIN {FEEDBACK_TABLE} f ON p.prediction_id = f.prediction_id
    WHERE f.feedback_id IS NULL
    ORDER BY p.predicted_at DESC
    LIMIT 10
""").collect()

print(f"Found {len(pending_feedback)} predictions awaiting feedback")
print("-" * 80)

# COMMAND ----------

# Automated batch feedback for demo
# In production, use the Streamlit app for interactive review
feedback_records = []

for pred in pending_feedback:
    # In real world, radiologist reviews X-ray and provides diagnosis
    # Here we simulate using the true_label we already have
    ground_truth = pred.actual_diagnosis
    ai_said = pred.ai_prediction

    # Determine feedback_type automatically
    if ai_said == 'PNEUMONIA' and ground_truth == 'PNEUMONIA':
        feedback_type = 'true-positive'
    elif ai_said == 'PNEUMONIA' and ground_truth == 'NORMAL':
        feedback_type = 'false-positive'
    elif ai_said == 'NORMAL' and ground_truth == 'NORMAL':
        feedback_type = 'true-negative'
    elif ai_said == 'NORMAL' and ground_truth == 'PNEUMONIA':
        feedback_type = 'false-negative'
    else:
        feedback_type = 'unknown'

    feedback_record = {
        'feedback_id': f"fb-{uuid.uuid4().hex[:12]}",
        'prediction_id': pred.prediction_id,
        'timestamp': datetime.now(),
        'ground_truth': ground_truth,
        'feedback_type': feedback_type,
        'radiologist_id': 'DR_DEMO_001',
        'confidence': 'confirmed',
        'feedback_source': 'demo_notebook',
        'notes': f'Simulated feedback for demo'
    }

    feedback_records.append(feedback_record)

    icon = "[OK]" if feedback_type.startswith('true') else "[ERR]"
    print(f"{icon} {pred.prediction_id[:16]}...")
    print(f"  AI: {ai_said} | Radiologist: {ground_truth}")
    print(f"  Type: {feedback_type}")
    print()

print(f"\nCollected {len(feedback_records)} feedback submissions")

# COMMAND ----------

# Write feedback to gold.prediction_feedback (Terraform table)
if feedback_records:
    feedback_df = spark.createDataFrame(feedback_records)

    feedback_df.write.mode('append').saveAsTable(FEEDBACK_TABLE)

    print(f"Wrote {len(feedback_records)} feedback records to {FEEDBACK_TABLE}")
else:
    print("No feedback to write")


# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **End-to-End Workflow Complete:**
# MAGIC 1. ✅ Endpoint warmed up and serving predictions
# MAGIC 2. ✅ A/B test predictions generated and logged to gold.pneumonia_predictions
# MAGIC 3. ✅ Inference logs captured for model tracking
# MAGIC 4. ✅ Radiologist feedback submitted to gold.prediction_feedback
# MAGIC
# MAGIC **Next Steps:**
# MAGIC - Use Streamlit app for interactive feedback collection: https://github.com/rgasiorek/healthcare-ai-platform-databricks/tree/main/apps/feedback_review
# MAGIC - View analytics in BI dashboard to compare model performance
# MAGIC - Check inference logs table to see which model (champion/challenger) served each request

# COMMAND ----------
