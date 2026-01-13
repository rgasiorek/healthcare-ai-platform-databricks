# Databricks notebook source
# MAGIC %md
# MAGIC # End-to-End MLOps Demo (30 Minutes)
# MAGIC
# MAGIC Complete workflow using Terraform-defined tables:
# MAGIC 1. Make predictions via A/B endpoint
# MAGIC 2. Submit radiologist feedback
# MAGIC 3. Monitor Champion vs Challenger performance
# MAGIC
# MAGIC **Tables Used** (from Terraform):
# MAGIC - `gold.pneumonia_predictions` (predictions from models)
# MAGIC - `gold.prediction_feedback` (ground truth from radiologists)

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
# MAGIC ## Step 1: Make Predictions (5-10 predictions)

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
# Helper: Preprocess image
import numpy as np
from PIL import Image

def preprocess_image(file_path, size=64):
    """Load and preprocess image for model input"""
    from io import BytesIO

    # Remove dbfs: prefix and use /dbfs/ for local file system access
    if file_path.startswith("dbfs:"):
        local_path = file_path.replace("dbfs:", "/dbfs")
    else:
        local_path = "/dbfs" + file_path if not file_path.startswith("/dbfs") else file_path

    # Unity Catalog Volumes are mounted at /Volumes, accessible via /dbfs/Volumes
    # Read binary file content
    with open(local_path, 'rb') as f:
        img_bytes = f.read()

    # Load image from bytes
    img = Image.open(BytesIO(img_bytes))
    img = img.convert('RGB')
    img = img.resize((size, size))
    img_array = np.array(img) / 255.0
    return img_array

# COMMAND ----------
# Make predictions and write to gold.pneumonia_predictions
predictions = []

for img in test_images:
    # Preprocess image
    img_array = preprocess_image(img.file_path)

    # Call A/B endpoint
    payload = {"inputs": [img_array.tolist()]}

    try:
        response = requests.post(
            invocation_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()
            pred_prob = result['predictions'][0][0]
            predicted_label = 1 if pred_prob > 0.5 else 0

            # Generate prediction_id
            prediction_id = f"pred-{uuid.uuid4().hex[:12]}"

            # Prepare record for our Terraform table
            prediction_record = {
                'prediction_id': prediction_id,
                'image_id': img.image_id,
                'predicted_label': predicted_label,
                'prediction_probability': float(pred_prob),
                'confidence_score': float(max(pred_prob, 1 - pred_prob)),
                'true_label': 1 if img.category == 'PNEUMONIA' else 0,  # We know true label from bronze
                'is_correct': (predicted_label == (1 if img.category == 'PNEUMONIA' else 0)),
                'model_name': ENDPOINT_NAME,  # A/B endpoint name
                'model_version': '1',
                'predicted_at': datetime.now(),
                'prediction_date': datetime.now().date()
            }

            predictions.append(prediction_record)

            match = "✅" if prediction_record['is_correct'] else "❌"
            print(f"{match} {img.filename}")
            print(f"   True: {img.category} | Predicted: {'PNEUMONIA' if predicted_label == 1 else 'NORMAL'}")
            print(f"   Probability: {pred_prob:.3f} | ID: {prediction_id}")
            print()

    except Exception as e:
        print(f"❌ Error predicting {img.filename}: {e}")

print(f"\n✅ Made {len(predictions)} predictions")

# COMMAND ----------
# Write predictions to gold.pneumonia_predictions (Terraform table)
if predictions:
    pred_df = spark.createDataFrame(predictions)

    pred_df.write.mode('append').saveAsTable(PREDICTIONS_TABLE)

    print(f"✅ Wrote {len(predictions)} predictions to {PREDICTIONS_TABLE}")
else:
    print("⚠️  No predictions to write")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Verify Predictions in Table

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

print(f"\n📊 Recent Predictions from {PREDICTIONS_TABLE}:")
print("=" * 120)
recent_predictions.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Submit Radiologist Feedback
# MAGIC
# MAGIC Simulate radiologist reviewing predictions and providing ground truth

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
# Submit feedback for each prediction
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

    icon = "✅" if feedback_type.startswith('true') else "❌"
    print(f"{icon} {pred.prediction_id[:16]}...")
    print(f"   AI: {ai_said} | Radiologist: {ground_truth}")
    print(f"   Type: {feedback_type}")
    print()

print(f"\n✅ Collected {len(feedback_records)} feedback submissions")

# COMMAND ----------
# Write feedback to gold.prediction_feedback (Terraform table)
if feedback_records:
    feedback_df = spark.createDataFrame(feedback_records)

    feedback_df.write.mode('append').saveAsTable(FEEDBACK_TABLE)

    print(f"✅ Wrote {len(feedback_records)} feedback records to {FEEDBACK_TABLE}")
else:
    print("⚠️  No feedback to write")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Monitor Performance - Champion vs Challenger

# COMMAND ----------
# Calculate accuracy by model
# Note: A/B endpoint doesn't expose which specific model (Keras vs PyTorch) served each request in our current setup
# For this demo, we'll show overall endpoint performance

performance = spark.sql(f"""
    SELECT
        model_name,
        COUNT(*) as total_predictions,
        SUM(CASE WHEN is_correct THEN 1 ELSE 0 END) as correct_predictions,
        ROUND(AVG(CASE WHEN is_correct THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_pct,
        COUNT(DISTINCT f.feedback_id) as feedback_count,
        ROUND(COUNT(DISTINCT f.feedback_id) * 100.0 / COUNT(*), 2) as feedback_coverage_pct
    FROM {PREDICTIONS_TABLE} p
    LEFT JOIN {FEEDBACK_TABLE} f ON p.prediction_id = f.prediction_id
    GROUP BY model_name
""")

print("\n📊 MODEL PERFORMANCE:")
print("=" * 80)
performance.show(truncate=False)

# COMMAND ----------
# Confusion Matrix (with feedback)
confusion = spark.sql(f"""
    SELECT
        f.feedback_type,
        COUNT(*) as count
    FROM {FEEDBACK_TABLE} f
    GROUP BY f.feedback_type
    ORDER BY f.feedback_type
""")

print("\n📊 CONFUSION MATRIX (From Feedback):")
print("=" * 80)
confusion.show(truncate=False)

# COMMAND ----------
# Detailed feedback analysis
feedback_analysis = spark.sql(f"""
    SELECT
        p.prediction_id,
        CASE WHEN p.predicted_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as ai_prediction,
        f.ground_truth,
        f.feedback_type,
        p.prediction_probability,
        f.radiologist_id,
        f.timestamp
    FROM {PREDICTIONS_TABLE} p
    INNER JOIN {FEEDBACK_TABLE} f ON p.prediction_id = f.prediction_id
    ORDER BY f.timestamp DESC
""")

print("\n📊 FEEDBACK ANALYSIS:")
print("=" * 80)
feedback_analysis.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Complete MLOps Workflow Demonstrated**:
# MAGIC
# MAGIC 1. ✅ Predictions made via A/B testing endpoint
# MAGIC 2. ✅ Data written to `gold.pneumonia_predictions` (Terraform table)
# MAGIC 3. ✅ Feedback collected (simulated radiologist review)
# MAGIC 4. ✅ Data written to `gold.prediction_feedback` (Terraform table)
# MAGIC 5. ✅ Performance monitored (accuracy, confusion matrix)
# MAGIC
# MAGIC **Tables Used** (All Terraform-defined):
# MAGIC - `healthcare_catalog_dev.gold.pneumonia_predictions`
# MAGIC - `healthcare_catalog_dev.gold.prediction_feedback`
# MAGIC
# MAGIC **Next Steps**:
# MAGIC - To see true A/B testing (Keras vs PyTorch), we need to capture `served_model_name` from endpoint
# MAGIC - For production, integrate feedback endpoint (REST API for radiologists)
# MAGIC - Build BI dashboard using these tables
