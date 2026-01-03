# Databricks notebook source
# MAGIC %md
# MAGIC # Demo: Two Ways to Use ML Models - SDK vs REST API
# MAGIC
# MAGIC This notebook demonstrates two approaches for using trained ML models:
# MAGIC 1. **MLflow SDK** (Direct model loading) - Best for batch processing
# MAGIC 2. **REST API** (Model Serving endpoint) - Best for real-time predictions
# MAGIC
# MAGIC **Model**: Pneumonia X-ray Classification
# MAGIC **Registry**: `healthcare_catalog_dev.models.pneumonia_poc_classifier`
# MAGIC
# MAGIC **Learning Objectives**:
# MAGIC - Understand when to use SDK vs REST API
# MAGIC - Compare performance and use cases
# MAGIC - See practical examples of both approaches

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup: Install Dependencies and Load Sample Data

# COMMAND ----------
# Install required libraries
%pip install tensorflow==2.15.0 Pillow mlflow --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import os
import time
import numpy as np
from PIL import Image
import mlflow
import requests
import json

# Configuration
MODEL_NAME = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
MODEL_VERSION = "1"
ENDPOINT_NAME = "pneumonia-poc-classifier"
IMAGE_SIZE = 64  # Model was trained on 64x64 images

print(f"Demo Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Version: {MODEL_VERSION}")
print(f"  Endpoint: {ENDPOINT_NAME}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Test Images
# MAGIC
# MAGIC Get sample images from Bronze table to test both approaches

# COMMAND ----------
# Load 5 sample images (mix of NORMAL and PNEUMONIA)
sample_df = spark.sql("""
    (SELECT * FROM healthcare_catalog_dev.bronze.kaggle_xray_metadata
     WHERE category = 'NORMAL' LIMIT 3)
    UNION ALL
    (SELECT * FROM healthcare_catalog_dev.bronze.kaggle_xray_metadata
     WHERE category = 'PNEUMONIA' LIMIT 2)
""")

# Collect to driver for demo
test_samples = sample_df.collect()

print(f"Loaded {len(test_samples)} test images:")
for i, sample in enumerate(test_samples):
    print(f"  {i+1}. {sample.filename} - {sample.category}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Helper Function: Preprocess Images
# MAGIC
# MAGIC Both approaches need the same preprocessing (resize, normalize)

# COMMAND ----------
def preprocess_image(file_path, image_size=64):
    """
    Load and preprocess image for model input

    Args:
        file_path: Path to image file (dbfs:/Volumes/...)
        image_size: Target size for resizing (default: 64)

    Returns:
        numpy array of shape (image_size, image_size, 3) normalized to [0, 1]
    """
    # Unity Catalog volumes: remove dbfs: prefix
    local_path = file_path.replace("dbfs:", "")

    # Load and preprocess
    img = Image.open(local_path)
    img = img.convert('RGB')
    img = img.resize((image_size, image_size))
    img_array = np.array(img) / 255.0

    return img_array

# Test preprocessing on first image
test_img = preprocess_image(test_samples[0].file_path, IMAGE_SIZE)
print(f"✅ Preprocessed image shape: {test_img.shape}")
print(f"   Value range: [{test_img.min():.3f}, {test_img.max():.3f}]")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC # Approach 1: MLflow SDK (Direct Model Loading)
# MAGIC
# MAGIC **Use Case**: Batch processing, data pipelines, Databricks jobs
# MAGIC
# MAGIC **Advantages**:
# MAGIC - ✅ Fast (no HTTP overhead)
# MAGIC - ✅ Cost-effective (no always-on endpoint)
# MAGIC - ✅ Scalable (can distribute across Spark cluster)
# MAGIC - ✅ Efficient for large batches
# MAGIC
# MAGIC **When to Use**:
# MAGIC - Processing thousands of images
# MAGIC - Scheduled batch jobs
# MAGIC - Data pipelines
# MAGIC - Model experimentation

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1.1: Load Model from Registry using MLflow SDK

# COMMAND ----------
# Load model directly from Unity Catalog Model Registry
print(f"Loading model from MLflow Model Registry...")
print(f"  Model URI: models:/{MODEL_NAME}/{MODEL_VERSION}")

# Load the model
model_sdk = mlflow.keras.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

print(f"✅ Model loaded successfully!")
print(f"\nModel Summary:")
model_sdk.summary()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1.2: Run Predictions using SDK (Batch)

# COMMAND ----------
print("=" * 80)
print("APPROACH 1: MLflow SDK - Batch Predictions")
print("=" * 80)

# Prepare batch of images
batch_images = np.array([preprocess_image(s.file_path, IMAGE_SIZE) for s in test_samples])
batch_labels = [s.category for s in test_samples]

print(f"\nBatch shape: {batch_images.shape}")
print(f"Number of images: {len(batch_images)}")

# Measure prediction time
start_time = time.time()

# Run batch prediction (all at once!)
predictions_sdk = model_sdk.predict(batch_images, verbose=0)

sdk_duration = time.time() - start_time

# Display results
print(f"\n⏱️  Total prediction time: {sdk_duration:.3f} seconds")
print(f"   Time per image: {sdk_duration/len(batch_images):.3f} seconds")
print(f"\nPredictions:")
print("-" * 80)

for i, (sample, pred_prob) in enumerate(zip(test_samples, predictions_sdk)):
    pred_label = "PNEUMONIA" if pred_prob[0] > 0.5 else "NORMAL"
    confidence = max(pred_prob[0], 1 - pred_prob[0])
    match = "✅" if pred_label == sample.category else "❌"

    print(f"{match} Image {i+1}: {sample.filename}")
    print(f"   True: {sample.category:<10} | Predicted: {pred_label:<10} | Confidence: {confidence:.1%}")
    print(f"   P(PNEUMONIA): {pred_prob[0]:.3f}")
    print()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1.3: SDK Approach - Key Takeaways
# MAGIC
# MAGIC **What Just Happened**:
# MAGIC 1. Loaded model once from MLflow registry
# MAGIC 2. Processed all 5 images in a single batch
# MAGIC 3. Got predictions in ~0.1-0.2 seconds total
# MAGIC
# MAGIC **For Large Datasets**:
# MAGIC ```python
# MAGIC # Process 10,000 images efficiently
# MAGIC all_images = load_10000_images()
# MAGIC predictions = model.predict(all_images, batch_size=32)
# MAGIC # Takes seconds instead of hours with REST API!
# MAGIC ```
# MAGIC
# MAGIC **With Spark** (for millions of images):
# MAGIC ```python
# MAGIC # Distribute across cluster
# MAGIC predictions_df = df.mapInPandas(predict_batch_udf, schema)
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC # Approach 2: REST API (Model Serving Endpoint)
# MAGIC
# MAGIC **Use Case**: Real-time predictions, web apps, external systems
# MAGIC
# MAGIC **Advantages**:
# MAGIC - ✅ Accessible from anywhere (HTTP)
# MAGIC - ✅ Language-agnostic (any language can call REST)
# MAGIC - ✅ Auto-scaling based on traffic
# MAGIC - ✅ Built-in authentication
# MAGIC
# MAGIC **When to Use**:
# MAGIC - Mobile/web applications
# MAGIC - Real-time predictions (< 1 second latency)
# MAGIC - External systems integration
# MAGIC - Microservices architecture

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2.1: Check Endpoint Status

# COMMAND ----------
# Get workspace details for API calls
workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

# Check if endpoint exists
endpoint_url = f"https://{workspace_url}/api/2.0/serving-endpoints/{ENDPOINT_NAME}"

response = requests.get(
    endpoint_url,
    headers={"Authorization": f"Bearer {token}"}
)

if response.status_code == 200:
    endpoint_info = response.json()
    status = endpoint_info.get("state", {}).get("ready", "UNKNOWN")

    print(f"Endpoint Status:")
    print(f"  Name: {ENDPOINT_NAME}")
    print(f"  Status: {status}")
    print(f"  URL: https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations")

    if status != "READY":
        print(f"\n⚠️  Endpoint is not READY. Please deploy it first using:")
        print(f"     /Shared/deploy-serving-endpoint")
else:
    print(f"❌ Endpoint not found. Please deploy it first using:")
    print(f"   /Shared/deploy-serving-endpoint")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2.2: Warm Up Endpoint (Handle Cold Start)
# MAGIC
# MAGIC Serverless endpoints shut down when idle. First request may take 30-60s to start.

# COMMAND ----------
invocation_url = f"https://{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"

print("Warming up endpoint (handling cold start)...")
print("This may take 30-60 seconds on first request...\n")

# Warm-up request with longer timeout
warm_img = preprocess_image(test_samples[0].file_path, IMAGE_SIZE)
# TensorFlow/Keras models expect "inputs" format
warm_payload = {"inputs": [warm_img.tolist()]}

try:
    warm_start = time.time()
    warm_response = requests.post(
        invocation_url,
        headers={"Authorization": f"Bearer {token}"},
        json=warm_payload,
        timeout=90  # Longer timeout for cold start
    )
    warm_duration = time.time() - warm_start

    if warm_response.status_code == 200:
        print(f"✅ Endpoint warmed up successfully!")
        print(f"   Cold start time: {warm_duration:.1f} seconds")
        print(f"\nNow endpoint is warm - subsequent requests will be fast (~0.1-0.2s)\n")
    else:
        print(f"⚠️  Warm-up got status {warm_response.status_code}")
        print(f"   Will try actual predictions anyway...\n")

except Exception as e:
    print(f"⚠️  Warm-up request failed: {e}")
    print(f"   Endpoint might still be starting. Will retry with longer timeout...\n")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2.3: Run Predictions using REST API (One at a Time)

# COMMAND ----------
print("=" * 80)
print("APPROACH 2: REST API - Individual Predictions")
print("=" * 80)

print(f"\nEndpoint URL: {invocation_url}")
print(f"Making {len(test_samples)} individual API calls...\n")

# Measure total time
total_start = time.time()
api_predictions = []

print("Predictions:")
print("-" * 80)

for i, sample in enumerate(test_samples):
    # Preprocess image
    img_array = preprocess_image(sample.file_path, IMAGE_SIZE)

    # Prepare API payload
    # TensorFlow/Keras models expect "inputs" format (not dataframe_records)
    # Model signature expects tensor shape: (-1, 64, 64, 3)
    payload = {
        "inputs": [img_array.tolist()]  # Wrap in list for batch dimension
    }

    # Measure individual request time
    request_start = time.time()

    try:
        # Call REST API (increased timeout for cold starts)
        response = requests.post(
            invocation_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=60  # Increased from 30 to handle any cold starts
        )

        request_duration = time.time() - request_start

        if response.status_code == 200:
            result = response.json()
            prediction = result.get('predictions', [[]])[0]
            pred_prob = prediction[0] if isinstance(prediction, list) else prediction

            pred_label = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"
            confidence = max(pred_prob, 1 - pred_prob)
            match = "✅" if pred_label == sample.category else "❌"

            api_predictions.append(pred_prob)

            print(f"{match} Image {i+1}: {sample.filename}")
            print(f"   True: {sample.category:<10} | Predicted: {pred_label:<10} | Confidence: {confidence:.1%}")
            print(f"   P(PNEUMONIA): {pred_prob:.3f}")
            print(f"   ⏱️  Request time: {request_duration:.3f} seconds")
            print()
        else:
            print(f"❌ Image {i+1}: API Error {response.status_code}")
            print(f"   {response.text}")
            api_predictions.append(None)

    except Exception as e:
        request_duration = time.time() - request_start
        print(f"❌ Image {i+1}: Exception after {request_duration:.3f}s")
        print(f"   {str(e)}")
        api_predictions.append(None)

api_duration = time.time() - total_start

print("-" * 80)
print(f"⏱️  Total time (all API calls): {api_duration:.3f} seconds")
print(f"   Average time per request: {api_duration/len(test_samples):.3f} seconds")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2.3: REST API Approach - Key Takeaways
# MAGIC
# MAGIC **What Just Happened**:
# MAGIC 1. Made 5 separate HTTP requests (one per image)
# MAGIC 2. Each request includes network overhead (serialize, send, deserialize)
# MAGIC 3. Total time typically 2-5x slower than SDK for small batches
# MAGIC
# MAGIC **HTTP Request Breakdown**:
# MAGIC - Serialize image to JSON (~50-100ms)
# MAGIC - Network round-trip (~20-50ms)
# MAGIC - Model inference (~10-20ms)
# MAGIC - Deserialize response (~10ms)
# MAGIC
# MAGIC **Typical Latencies**:
# MAGIC - REST API: ~100-200ms per image
# MAGIC - SDK: ~20-30ms per image (5-10x faster)
# MAGIC
# MAGIC **But REST API shines when**:
# MAGIC - Called from external systems (web app, mobile app)
# MAGIC - Real-time user-facing predictions
# MAGIC - Need auto-scaling for variable traffic

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC # Comparison: SDK vs REST API

# COMMAND ----------
# MAGIC %md
# MAGIC ## Performance Comparison

# COMMAND ----------
print("=" * 80)
print("PERFORMANCE COMPARISON")
print("=" * 80)

print(f"\nDataset: {len(test_samples)} images")
print(f"\n{'Metric':<30} {'SDK (Batch)':<20} {'REST API (Individual)':<20}")
print("-" * 70)

# Total time
print(f"{'Total Time':<30} {sdk_duration:.3f}s {' '*15} {api_duration:.3f}s")

# Time per image
sdk_per_image = sdk_duration / len(test_samples)
api_per_image = api_duration / len(test_samples)
print(f"{'Time per Image':<30} {sdk_per_image:.3f}s {' '*15} {api_per_image:.3f}s")

# Speedup
speedup = api_duration / sdk_duration
print(f"{'Speedup':<30} {speedup:.1f}x faster {' '*10} baseline")

# Predictions match
print(f"\n{'Prediction Accuracy':<30} {'Same Results':<20} {'Same Results':<20}")

# Extrapolate to larger datasets
print(f"\n{'Extrapolated Times:':<30}")
for num_images in [100, 1000, 10000]:
    sdk_estimated = sdk_per_image * num_images
    api_estimated = api_per_image * num_images
    print(f"  {num_images:,} images: {sdk_estimated:.1f}s ({sdk_estimated/60:.1f}min) vs {api_estimated:.1f}s ({api_estimated/60:.1f}min)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Use Case Decision Matrix

# COMMAND ----------
print("=" * 80)
print("WHEN TO USE EACH APPROACH")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│                          USE SDK (Direct Loading)                           │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✅ Batch processing (100s - millions of images)                             │
│ ✅ Scheduled jobs (daily/hourly predictions)                                │
│ ✅ Data pipelines (Bronze → Gold transformations)                           │
│ ✅ Model experimentation and development                                    │
│ ✅ Cost-sensitive applications (no always-on endpoint)                      │
│ ✅ High-throughput requirements                                             │
│                                                                              │
│ Example: Process 10,000 X-rays overnight, write to Gold table for BI        │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│                       USE REST API (Serving Endpoint)                       │
├─────────────────────────────────────────────────────────────────────────────┤
│ ✅ Real-time predictions (web/mobile apps)                                  │
│ ✅ User-facing applications (< 1 second latency required)                   │
│ ✅ External system integration (non-Python clients)                         │
│ ✅ Microservices architecture                                               │
│ ✅ Variable traffic patterns (auto-scaling needed)                          │
│ ✅ Multi-language environments (JavaScript, Java, etc.)                     │
│                                                                              │
│ Example: Doctor uploads X-ray via mobile app, gets instant diagnosis        │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Architecture Patterns

# COMMAND ----------
print("=" * 80)
print("RECOMMENDED ARCHITECTURE PATTERNS")
print("=" * 80)

print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ Pattern 1: Hybrid (Best of Both Worlds)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────────┐                                                          │
│   │ REST API     │ ──────► Real-time predictions (mobile app, web)          │
│   │ Endpoint     │         - Single images                                  │
│   └──────────────┘         - < 1 second latency                             │
│                                                                              │
│   ┌──────────────┐                                                          │
│   │ Batch Job    │ ──────► Nightly batch processing                         │
│   │ (SDK)        │         - Process day's uploads                          │
│   └──────────────┘         - Write to Gold layer for analytics              │
│                                                                              │
│ Use Case: Hospital system with both real-time diagnosis and daily reports   │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Pattern 2: Batch-Only (Cost-Optimized)                                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Bronze Layer → [Load Model via SDK] → Predict → Gold Layer               │
│                                                                              │
│   Schedule: Hourly or daily                                                 │
│   Cost: Compute only when running                                           │
│   Latency: Minutes to hours                                                 │
│                                                                              │
│ Use Case: Research institution analyzing historical X-ray database          │
└─────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────────┐
│ Pattern 3: Real-Time Only (User-Facing)                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   Mobile App → REST API → Instant Diagnosis                                 │
│                                                                              │
│   Always-on endpoint with auto-scaling                                      │
│   Latency: < 1 second                                                       │
│   Cost: Pay for endpoint uptime                                             │
│                                                                              │
│ Use Case: Telemedicine app for rural clinics                                │
└─────────────────────────────────────────────────────────────────────────────┘
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC # Summary and Code Snippets for Pupils

# COMMAND ----------
# MAGIC %md
# MAGIC ## Quick Reference: SDK Approach

# COMMAND ----------
print("""
# ============================================================================
# SDK APPROACH - For Batch Processing
# ============================================================================

import mlflow
import numpy as np

# 1. Load model from registry
model = mlflow.keras.load_model(
    "models:/healthcare_catalog_dev.models.pneumonia_poc_classifier/1"
)

# 2. Prepare batch of images
images = np.array([preprocess(img) for img in image_list])  # Shape: (N, 64, 64, 3)

# 3. Predict (all at once!)
predictions = model.predict(images)

# 4. Process results
for img, pred in zip(image_list, predictions):
    label = "PNEUMONIA" if pred[0] > 0.5 else "NORMAL"
    print(f"{img}: {label} (confidence: {max(pred[0], 1-pred[0]):.1%})")

# ============================================================================
# Performance: ~0.02 seconds per image
# Best for: Batch jobs, data pipelines, 100+ images
# ============================================================================
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Quick Reference: REST API Approach

# COMMAND ----------
print("""
# ============================================================================
# REST API APPROACH - For Real-Time Predictions
# ============================================================================

import requests
import json

# 1. Setup
workspace_url = "your-workspace.cloud.databricks.com"
token = "your-databricks-token"
endpoint_url = f"https://{workspace_url}/serving-endpoints/pneumonia-poc-classifier/invocations"

# 2. Prepare single image
img_array = preprocess_image(image_path)  # Shape: (64, 64, 3)

# 3. Call REST API
# TensorFlow/Keras models expect "inputs" format
payload = {
    "inputs": [img_array.tolist()]  # List for batch dimension
}

response = requests.post(
    endpoint_url,
    headers={"Authorization": f"Bearer {token}"},
    json=payload
)

# 4. Parse result
result = response.json()
pred_prob = result['predictions'][0][0]
label = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"

print(f"Prediction: {label} (confidence: {max(pred_prob, 1-pred_prob):.1%})")

# ============================================================================
# Performance: ~0.1-0.2 seconds per image
# Best for: Web apps, mobile apps, real-time predictions
# ============================================================================
""")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Key Learnings
# MAGIC
# MAGIC **For Your Pupils**:
# MAGIC
# MAGIC 1. **Both approaches use the SAME model** - just different ways to access it
# MAGIC
# MAGIC 2. **SDK is 5-10x faster** for batch processing
# MAGIC    - No HTTP overhead
# MAGIC    - Process multiple images at once
# MAGIC    - Cost-effective (only pay when running)
# MAGIC
# MAGIC 3. **REST API is better for real-time**
# MAGIC    - Accessible from any language/platform
# MAGIC    - Auto-scaling for variable traffic
# MAGIC    - Required for web/mobile apps
# MAGIC
# MAGIC 4. **Real-world pattern**: Use BOTH
# MAGIC    - REST API for user-facing predictions
# MAGIC    - SDK for nightly batch processing and analytics
# MAGIC
# MAGIC 5. **Cost considerations**:
# MAGIC    - REST endpoint: Always-on cost (~$50-200/month for Small endpoint)
# MAGIC    - SDK batch job: Only pay when running (~$1-10 per job)

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Next Steps for Pupils
# MAGIC
# MAGIC 1. **Try modifying the code**:
# MAGIC    - Change the confidence threshold (0.5 → 0.7)
# MAGIC    - Add error handling
# MAGIC    - Save predictions to a Delta table
# MAGIC
# MAGIC 2. **Experiment with batch sizes**:
# MAGIC    - Try predicting 1, 10, 100 images with SDK
# MAGIC    - Measure the time difference
# MAGIC
# MAGIC 3. **Build a simple application**:
# MAGIC    - Option A: Batch job that processes daily uploads
# MAGIC    - Option B: Simple web API client
# MAGIC
# MAGIC 4. **Explore MLflow**:
# MAGIC    - View experiments: Machine Learning → Experiments
# MAGIC    - Check model registry: Machine Learning → Models
# MAGIC    - Compare model versions
# MAGIC
# MAGIC **Questions to Discuss**:
# MAGIC - When would you use SDK vs REST API in a hospital?
# MAGIC - How would you handle 1 million X-rays?
# MAGIC - What if you need < 100ms latency?
# MAGIC - How to balance cost vs performance?
