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
MODEL_NAME = "healthcare_catalog_dev.models.pneumonia_poc_classifier_remote_file"
MODEL_VERSION = "9"  # Path-based model (Files API)
ENDPOINT_NAME = "pneumonia-poc-classifier"  # Simple demo endpoint (not A/B testing)
IMAGE_SIZE = 64  # Model was trained on 64x64 images

print(f"Demo Configuration:")
print(f"  Model: {MODEL_NAME}")
print(f"  Version: {MODEL_VERSION}")
print(f"  Endpoint: {ENDPOINT_NAME}")
print(f"\n📝 Note: Using v9 path-based models (Files API)")

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
# MAGIC ## Note: Path-Based Models (v9)
# MAGIC
# MAGIC V9 models use **Files API** and accept Unity Catalog file paths instead of image arrays.
# MAGIC
# MAGIC **Input Format**:
# MAGIC - Old (v1): numpy array of image pixels
# MAGIC - New (v9): `{"file_path": "/Volumes/catalog/schema/volume/image.jpeg"}`
# MAGIC
# MAGIC **Advantage**: No need to load/preprocess images - model does it internally using WorkspaceClient

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

# Load the PyFunc model (v9 is a custom wrapper)
model_sdk = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}/{MODEL_VERSION}")

print(f"✅ Model loaded successfully!")
print(f"   Type: {type(model_sdk)}")
print(f"   Metadata: {model_sdk.metadata}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1.2: Run Predictions using SDK (Batch)

# COMMAND ----------
print("=" * 80)
print("APPROACH 1: MLflow SDK - Batch Predictions")
print("=" * 80)

# Prepare batch of file paths (v9 models accept paths, not arrays)
import pandas as pd

batch_data = pd.DataFrame([
    {"file_path": s.file_path} for s in test_samples
])
batch_labels = [s.category for s in test_samples]

print(f"\nBatch size: {len(batch_data)} images")
print(f"Input format: file_path (Unity Catalog volumes)")

# Measure prediction time
start_time = time.time()

# Run batch prediction with file paths
predictions_sdk = model_sdk.predict(batch_data)

sdk_duration = time.time() - start_time

# Display results
print(f"\n⏱️  Total prediction time: {sdk_duration:.3f} seconds")
print(f"   Time per image: {sdk_duration/len(batch_data):.3f} seconds")
print(f"\nPredictions:")
print("-" * 80)

for i, (sample, pred) in enumerate(zip(test_samples, predictions_sdk)):
    # v9 model returns dict with 'prediction' and 'probability'
    pred_label = pred['prediction']
    pred_prob = pred['probability']
    confidence = pred_prob if pred_label == "PNEUMONIA" else (1 - pred_prob)
    match = "✅" if pred_label == sample.category else "❌"

    print(f"{match} Image {i+1}: {sample.filename}")
    print(f"   True: {sample.category:<10} | Predicted: {pred_label:<10} | Confidence: {confidence:.1%}")
    print(f"   P(PNEUMONIA): {pred_prob:.3f}")
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

# Warm-up request with file_path (v9 model format)
warm_payload = {
    "inputs": [{"file_path": test_samples[0].file_path}]
}

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
        print(f"\nNow endpoint is warm - subsequent requests will be fast (~1-2s)\n")
    else:
        print(f"⚠️  Warm-up got status {warm_response.status_code}")
        print(f"   Response: {warm_response.text}")
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
    # Prepare API payload with file_path (v9 model format)
    payload = {
        "inputs": [{"file_path": sample.file_path}]
    }

    # Measure individual request time
    request_start = time.time()

    try:
        # Call REST API
        response = requests.post(
            invocation_url,
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
            timeout=60
        )

        request_duration = time.time() - request_start

        if response.status_code == 200:
            result = response.json()
            # v9 model returns dict with 'prediction' and 'probability'
            pred = result.get('predictions', [{}])[0]
            pred_label = pred['prediction']
            pred_prob = pred['probability']

            confidence = pred_prob if pred_label == "PNEUMONIA" else (1 - pred_prob)
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

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## NEW: Prediction Tracking & Feedback Loop
# MAGIC
# MAGIC In production, we need to:
# MAGIC 1. **Track which predictions we made** (so we can link feedback later)
# MAGIC 2. **Collect ground truth labels** (from radiologists/doctors)
# MAGIC 3. **Link feedback back to the model** that made the prediction
# MAGIC 4. **Measure real-world accuracy** and compare models
# MAGIC
# MAGIC This is especially critical for **A/B testing** when multiple models serve the same endpoint!

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Capturing Prediction IDs from REST API
# MAGIC
# MAGIC Every Databricks Model Serving request gets a unique `request_id` that we can use to link feedback.

# COMMAND ----------
# Make a prediction and capture the request_id
test_image_path = test_samples[0].file_path
img_array = preprocess_image(test_image_path, IMAGE_SIZE)

payload = {"inputs": [img_array.tolist()]}

response = requests.post(
    invocation_url,
    headers={"Authorization": f"Bearer {token}"},
    json=payload
)

if response.status_code == 200:
    # Get prediction
    result = response.json()
    pred_prob = result['predictions'][0][0]
    pred_label = "PNEUMONIA" if pred_prob > 0.5 else "NORMAL"

    # CRITICAL: Capture the request_id from response headers
    request_id = response.headers.get('x-databricks-request-id')

    print("✅ Prediction Made:")
    print(f"   Request ID: {request_id}")
    print(f"   Predicted: {pred_label}")
    print(f"   Confidence: {max(pred_prob, 1-pred_prob):.1%}")
    print(f"   P(PNEUMONIA): {pred_prob:.3f}")
    print()
    print("💡 SAVE THIS REQUEST_ID! You'll need it to submit feedback later.")
    print(f"   In a real app, you would: display_to_user(prediction, request_id)")
else:
    print(f"❌ Prediction failed: {response.status_code}")
    print(response.text)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Submitting Feedback (After Radiologist Review)
# MAGIC
# MAGIC Later, when a radiologist reviews the X-ray and provides ground truth, we submit feedback.

# COMMAND ----------
# Install feedback collector (if not already available)
import sys
sys.path.append('/Workspace/Shared')

# Import the feedback collector
try:
    from feedback_collector import submit_feedback
    print("✅ Feedback collector loaded")
except ImportError:
    print("⚠️  Feedback collector not found. Upload it first using:")
    print("   /tmp/upload_feedback_collector.py")

# COMMAND ----------
# Simulate: Radiologist reviews the X-ray and confirms diagnosis
# In real app, this would happen hours/days later

# Example 1: Model was CORRECT
# request_id from earlier: 'abc-123-def-456'
# Radiologist confirms: TRUE PNEUMONIA

feedback_id_1 = submit_feedback(
    prediction_id=request_id,  # The request_id we captured earlier
    feedback_type="true-positive",  # Model said PNEUMONIA, it WAS pneumonia
    radiologist_id="DR001",
    confidence="confirmed",
    feedback_source="expert_review",
    notes="Clear consolidation in right lower lobe"
)

print(f"✅ Feedback submitted: {feedback_id_1}")
print(f"   Prediction: {request_id} → Ground Truth: PNEUMONIA")

# COMMAND ----------
# Example 2: Model was WRONG (False Positive)
# Make another prediction first
img_array_2 = preprocess_image(test_samples[1].file_path, IMAGE_SIZE)
response_2 = requests.post(
    invocation_url,
    headers={"Authorization": f"Bearer {token}"},
    json={"inputs": [img_array_2.tolist()]}
)

if response_2.status_code == 200:
    request_id_2 = response_2.headers.get('x-databricks-request-id')
    pred_prob_2 = response_2.json()['predictions'][0][0]
    pred_label_2 = "PNEUMONIA" if pred_prob_2 > 0.5 else "NORMAL"

    print(f"Prediction 2: {pred_label_2} (request_id: {request_id_2})")

    # Radiologist says: "Actually, this was NORMAL (false positive)"
    if pred_label_2 == "PNEUMONIA":
        feedback_id_2 = submit_feedback(
            prediction_id=request_id_2,
            feedback_type="false-positive",  # Model said PNEUMONIA, actually NORMAL
            radiologist_id="DR001",
            confidence="confirmed",
            notes="Artifact from patient movement, not infection"
        )
        print(f"✅ Feedback (false positive): {feedback_id_2}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Understanding A/B Testing - Which Model Served?
# MAGIC
# MAGIC When using A/B testing endpoints (Champion/Challenger), Databricks automatically logs:
# MAGIC - Which model served each request
# MAGIC - The request_id
# MAGIC - The prediction
# MAGIC
# MAGIC This is stored in the **inference table**.

# COMMAND ----------
# Query the inference table to see which model served our prediction
from pyspark.sql.functions import col

# Databricks auto-creates inference tables when auto_capture is enabled
# Format: {catalog}.{schema}.{table_prefix}_payload and {table_prefix}_predictions
inference_table = "healthcare_catalog_dev.gold.pneumonia_classifier_predictions"

try:
    # Check if our request_id is logged (may take 1-2 minutes for data to appear)
    inference_df = spark.sql(f"""
        SELECT
            request_id,
            served_model_name,  -- CRITICAL: Which model served this request
            timestamp,
            response
        FROM {inference_table}
        WHERE request_id = '{request_id}'
    """)

    if inference_df.count() > 0:
        model_info = inference_df.collect()[0]
        print("✅ Found prediction in inference table:")
        print(f"   Request ID: {model_info.request_id}")
        print(f"   Model: {model_info.served_model_name}")
        print(f"   Timestamp: {model_info.timestamp}")
        print()
        print("💡 This tells us WHICH MODEL (Champion or Challenger) made this prediction!")
    else:
        print("⏳ Prediction not yet in inference table (data appears within 1-2 minutes)")
        print(f"   Check later with: SELECT * FROM {inference_table} WHERE request_id = '{request_id}'")

except Exception as e:
    print(f"⚠️  Could not query inference table: {e}")
    print(f"   Make sure A/B testing endpoint has auto_capture enabled")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: The Complete Feedback Loop
# MAGIC
# MAGIC Now we can JOIN predictions with feedback to calculate per-model accuracy:

# COMMAND ----------
# Query model performance (joins inference table with feedback)
performance_query = """
SELECT
    p.request_id,
    p.served_model_name,                          -- Which model made the prediction
    CAST(p.response:predictions[0][0] AS DOUBLE) as prediction_score,
    CASE
        WHEN CAST(p.response:predictions[0][0] AS DOUBLE) > 0.5
        THEN 'PNEUMONIA'
        ELSE 'NORMAL'
    END as predicted_label,
    f.ground_truth,                                -- What it actually was (from radiologist)
    f.feedback_type,                               -- TP/FP/TN/FN
    CASE
        WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN TRUE
        WHEN f.feedback_type IN ('false-positive', 'false-negative') THEN FALSE
        ELSE NULL
    END as is_correct,
    f.radiologist_id,
    f.confidence,
    p.timestamp as prediction_time,
    f.timestamp as feedback_time
FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
    ON p.request_id = f.prediction_id
WHERE f.feedback_id IS NOT NULL  -- Only predictions with feedback
ORDER BY p.timestamp DESC
LIMIT 10
"""

try:
    performance_df = spark.sql(performance_query)

    print("=" * 100)
    print("PREDICTIONS WITH FEEDBACK (Most Recent)")
    print("=" * 100)

    if performance_df.count() > 0:
        performance_df.show(truncate=False)

        # Calculate accuracy per model
        accuracy_query = """
        SELECT
            served_model_name,
            COUNT(*) as total_predictions_with_feedback,
            SUM(CASE WHEN is_correct = TRUE THEN 1 ELSE 0 END) as correct_predictions,
            ROUND(AVG(CASE WHEN is_correct = TRUE THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_pct
        FROM (
            SELECT
                p.served_model_name,
                CASE
                    WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN TRUE
                    WHEN f.feedback_type IN ('false-positive', 'false-negative') THEN FALSE
                    ELSE NULL
                END as is_correct
            FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
            LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
                ON p.request_id = f.prediction_id
            WHERE f.feedback_id IS NOT NULL
        )
        GROUP BY served_model_name
        ORDER BY accuracy_pct DESC
        """

        accuracy_df = spark.sql(accuracy_query)

        print("\n" + "=" * 100)
        print("MODEL ACCURACY COMPARISON (Based on Real Feedback)")
        print("=" * 100)
        accuracy_df.show(truncate=False)

        print("\n💡 This is how we decide which model to promote:")
        print("   - Champion: Current production model")
        print("   - Challenger: New model being tested")
        print("   - If Challenger accuracy > Champion → Promote Challenger")
        print("   - If Challenger accuracy < Champion → Keep Champion")

    else:
        print("No predictions with feedback yet.")
        print("Submit some feedback using the examples above!")

except Exception as e:
    print(f"⚠️  Error querying performance: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Practical Exercise for Pupils
# MAGIC
# MAGIC **Task**: Implement a complete feedback tracking system
# MAGIC
# MAGIC 1. **Make 10 predictions** via REST API
# MAGIC    - Save all request_ids to a list
# MAGIC    - Print predictions
# MAGIC
# MAGIC 2. **Simulate radiologist review**
# MAGIC    - For each prediction, submit feedback
# MAGIC    - Mix of correct (TP/TN) and incorrect (FP/FN)
# MAGIC
# MAGIC 3. **Calculate model accuracy**
# MAGIC    - Query the performance view
# MAGIC    - Calculate: correct / total
# MAGIC
# MAGIC 4. **Visualize results**
# MAGIC    - Create a confusion matrix
# MAGIC    - Plot accuracy over time
# MAGIC
# MAGIC **Bonus Challenge**:
# MAGIC - Build a simple function that takes a request_id and ground_truth
# MAGIC - Automatically determines feedback_type (TP/FP/TN/FN)
# MAGIC - Submits feedback with proper classification
# MAGIC
# MAGIC **Discussion Questions**:
# MAGIC - How long should we collect feedback before promoting a Challenger?
# MAGIC - What if Champion has 95% accuracy and Challenger has 96%? (Is 1% improvement significant?)
# MAGIC - How to handle cases where radiologists disagree?
# MAGIC - What if we have multiple radiologists with different accuracy rates?

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Summary: Complete MLOps Cycle
# MAGIC
# MAGIC We've now covered the **complete production ML workflow**:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────┐
# MAGIC │                   COMPLETE MLOPS CYCLE                          │
# MAGIC └─────────────────────────────────────────────────────────────────┘
# MAGIC
# MAGIC 1. TRAIN
# MAGIC    └─► Train models (Keras Champion + PyTorch Challenger)
# MAGIC        Register in MLflow Model Registry
# MAGIC
# MAGIC 2. DEPLOY
# MAGIC    └─► Create A/B testing endpoint
# MAGIC        Champion: 90% traffic
# MAGIC        Challenger: 10% traffic
# MAGIC        Enable inference logging (auto_capture)
# MAGIC
# MAGIC 3. PREDICT
# MAGIC    └─► REST API: Make predictions
# MAGIC        Capture request_id from response headers
# MAGIC        Return prediction + request_id to user
# MAGIC
# MAGIC 4. COLLECT FEEDBACK
# MAGIC    └─► Radiologist reviews X-ray (hours/days later)
# MAGIC        submit_feedback(request_id, "true-positive", ...)
# MAGIC        Stored in prediction_feedback table
# MAGIC
# MAGIC 5. ANALYZE
# MAGIC    └─► JOIN inference_table + feedback_table
# MAGIC        Calculate per-model accuracy
# MAGIC        Champion: 92.3% accurate
# MAGIC        Challenger: 94.7% accurate
# MAGIC
# MAGIC 6. DECIDE & PROMOTE
# MAGIC    └─► Challenger is better! Promote to Champion
# MAGIC        Update traffic: Challenger 90%, New_Challenger 10%
# MAGIC        Continuous improvement!
# MAGIC
# MAGIC 7. MONITOR
# MAGIC    └─► Track performance over time
# MAGIC        Detect model drift
# MAGIC        Alert if accuracy drops
# MAGIC ```
# MAGIC
# MAGIC **Key Databricks Features Used**:
# MAGIC - ✅ MLflow Model Registry (versioning)
# MAGIC - ✅ Model Serving (REST API endpoints)
# MAGIC - ✅ A/B Testing (traffic splitting)
# MAGIC - ✅ Inference Tables (auto-capture predictions)
# MAGIC - ✅ Delta Tables (feedback storage with ACID)
# MAGIC - ✅ Unity Catalog (governance)
# MAGIC
# MAGIC **This is production-ready MLOps!** 🎉
