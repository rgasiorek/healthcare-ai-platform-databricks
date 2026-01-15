# Databricks notebook source
# MAGIC %md
# MAGIC # Generate Sample Predictions via A/B Testing Endpoint
# MAGIC
# MAGIC Calls the A/B testing endpoint to create predictions from both models (Keras and PyTorch).
# MAGIC
# MAGIC **What this does**:
# MAGIC - Loads 10-20 images from bronze layer
# MAGIC - Calls the A/B endpoint for each image
# MAGIC - Writes predictions to `gold.pneumonia_predictions`
# MAGIC - With 50/50 traffic split, both models will be used
# MAGIC
# MAGIC **Usage**: Run this notebook to quickly populate predictions table for testing feedback workflow.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------
import requests
import uuid
import numpy as np
from datetime import datetime
from PIL import Image
from io import BytesIO
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, DoubleType, BooleanType, TimestampType, DateType

# Configuration
CATALOG = "healthcare_catalog_dev"
SCHEMA_BRONZE = "bronze"
SCHEMA_GOLD = "gold"
PREDICTIONS_TABLE = f"{CATALOG}.{SCHEMA_GOLD}.pneumonia_predictions"

# A/B Testing Endpoint
ENDPOINT_NAME = "pneumonia-classifier-ab-test"

# Get workspace credentials
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
invocation_url = f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"

print("Configuration:")
print(f"  Endpoint: {ENDPOINT_NAME}")
print(f"  Predictions Table: {PREDICTIONS_TABLE}")
print(f"  URL: {invocation_url}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Warm Up Endpoint (First Request)
# MAGIC
# MAGIC **Cost savings & Security**: Endpoint configured with `scale_to_zero_enabled = true`
# MAGIC - No charges when idle
# MAGIC - Endpoint shut down when not in use
# MAGIC - First request takes 2-3 minutes (cold start)

# COMMAND ----------
print("🔥 Warming up endpoint...")
print("   Endpoint configured with scale_to_zero_enabled = true")
print("   → Cost savings: no charges when idle")
print("   → Security: endpoint shut down when not in use")
print("   Waiting for cold start (may take 2-3 minutes)...")

# Send dummy request to warm up
dummy_data = np.random.rand(64, 64, 3).tolist()
warmup_payload = {"inputs": [dummy_data]}

try:
    warmup_response = requests.post(
        invocation_url,
        headers={"Authorization": f"Bearer {token}"},
        json=warmup_payload,
        timeout=300  # 5 minutes for cold start
    )
    if warmup_response.status_code == 200:
        print("✅ Endpoint is warm and ready!")
    else:
        print(f"⚠️  Warm-up returned status {warmup_response.status_code}")
except Exception as e:
    print(f"❌ Warm-up failed: {e}")
    print("   Try running this cell again in a few minutes")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Sample Images from Bronze

# COMMAND ----------
# Get sample images with file paths
sample_images = spark.sql(f"""
    SELECT image_id, filename, category, file_path
    FROM {CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata
    LIMIT 20
""").collect()

print(f"Loaded {len(sample_images)} sample images from bronze layer")
print("-" * 80)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Helper: Preprocess Images

# COMMAND ----------
def preprocess_image(file_path, size=64):
    """Load and preprocess image for model input"""
    # Convert path format if needed
    if file_path.startswith("dbfs:"):
        volume_path = file_path.replace("dbfs:", "")
    else:
        volume_path = file_path

    # Read file using Spark binaryFiles (works with Unity Catalog Volumes)
    binary_df = spark.read.format("binaryFile").load(volume_path)
    file_content = binary_df.select("content").collect()[0][0]

    # Load image from bytes
    img = Image.open(BytesIO(file_content))
    img = img.convert('RGB')
    img = img.resize((size, size))
    img_array = np.array(img) / 255.0
    return img_array

# COMMAND ----------
# MAGIC %md
# MAGIC ## Make Predictions via A/B Testing Endpoint
# MAGIC
# MAGIC Calls the endpoint for each image. With 50/50 traffic split:
# MAGIC - ~10 predictions → Both models will be used
# MAGIC - ~20 predictions → ~10 per model (statistically balanced)

# COMMAND ----------
# Make predictions via A/B endpoint
predictions = []
first_request = True

for img in sample_images:
    try:
        # Preprocess image
        img_array = preprocess_image(img.file_path)

        # Call A/B endpoint
        payload = {"inputs": [img_array.tolist()]}

        # First request may take longer (already warmed up, but give extra time)
        timeout = 180 if first_request else 60

        response = requests.post(
            invocation_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=timeout
        )

        first_request = False

        if response.status_code == 200:
            result = response.json()
            pred_prob = float(result['predictions'][0][0])
            predicted_label = 1 if pred_prob > 0.5 else 0

            # Generate prediction ID
            prediction_id = f"pred-{uuid.uuid4().hex[:12]}"

            # Calculate confidence
            confidence_score = pred_prob if pred_prob > 0.5 else (1 - pred_prob)

            # True label
            true_label = 1 if img.category == 'PNEUMONIA' else 0

            # Prepare record
            prediction_record = {
                'prediction_id': prediction_id,
                'image_id': img.image_id,
                'predicted_label': predicted_label,
                'prediction_probability': pred_prob,
                'confidence_score': confidence_score,
                'true_label': true_label,
                'is_correct': (predicted_label == true_label),
                'model_name': ENDPOINT_NAME,  # A/B endpoint name
                'model_version': '1',
                'predicted_at': datetime.now(),
                'prediction_date': datetime.now().date()
            }

            predictions.append(prediction_record)

            print(f"✅ {img.filename[:30]:<30} | Predicted: {'PNEUMONIA' if predicted_label == 1 else 'NORMAL':<10} | Probability: {pred_prob:.3f}")

        else:
            print(f"❌ Error {response.status_code} for {img.filename}: {response.text}")

    except Exception as e:
        print(f"❌ Error predicting {img.filename}: {e}")

print(f"\n✅ Made {len(predictions)} predictions via A/B endpoint")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Write to Gold Layer

# COMMAND ----------
# Define schema matching Terraform table
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

# Create DataFrame
pred_df = spark.createDataFrame(predictions, schema=schema)

# Write to gold layer
pred_df.write.mode('append').saveAsTable(PREDICTIONS_TABLE)

print(f"✅ Wrote {len(predictions)} predictions to {PREDICTIONS_TABLE}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Verify Results

# COMMAND ----------
# Query recent predictions
recent = spark.sql(f"""
    SELECT
        prediction_id,
        image_id,
        CASE WHEN predicted_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as predicted_diagnosis,
        ROUND(prediction_probability, 3) as probability,
        model_name,
        predicted_at
    FROM {PREDICTIONS_TABLE}
    ORDER BY predicted_at DESC
    LIMIT 20
""")

print("\n📋 Recent Predictions (awaiting radiologist feedback):")
print("=" * 120)
recent.show(truncate=False)

# COMMAND ----------
# Count predictions by model (if available in future - currently all show same endpoint name)
model_distribution = spark.sql(f"""
    SELECT
        model_name,
        COUNT(*) as prediction_count
    FROM {PREDICTIONS_TABLE}
    GROUP BY model_name
    ORDER BY prediction_count DESC
""")

print("\n📊 Model Distribution:")
print("=" * 80)
model_distribution.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ✅ **Generated predictions via A/B testing endpoint**
# MAGIC
# MAGIC **What was created:**
# MAGIC - 10-20 predictions from real models (Keras and PyTorch)
# MAGIC - With 50/50 traffic split, both models served requests
# MAGIC - All predictions written to `gold.pneumonia_predictions`
# MAGIC - **Status**: Awaiting radiologist feedback
# MAGIC
# MAGIC **Next Steps - Submit Feedback:**
# MAGIC 1. **Streamlit App** (recommended): Run `streamlit run app.py` for interactive table-based feedback
# MAGIC 2. **Notebook**: Use `end-to-end-demo` notebook feedback section
# MAGIC
# MAGIC **To generate more predictions**: Run this notebook again (appends to table)
# MAGIC
# MAGIC **Note**: First run takes 2-3 minutes for endpoint warm-up (scale-to-zero cost savings).
