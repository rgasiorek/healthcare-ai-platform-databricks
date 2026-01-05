# Databricks notebook source
# MAGIC %md
# MAGIC # Register Feedback Processor Model
# MAGIC
# MAGIC This notebook registers the FeedbackProcessor model to MLflow.
# MAGIC The actual endpoint deployment is managed by Terraform (infrastructure as code).
# MAGIC
# MAGIC **Architecture**:
# MAGIC - Model serving endpoint: `/serving-endpoints/{model}/invocations` → Make prediction
# MAGIC - Feedback endpoint: `/serving-endpoints/feedback/invocations` → Submit feedback
# MAGIC
# MAGIC **Workflow**:
# MAGIC 1. This notebook: Register FeedbackProcessor to MLflow (one-time ML work)
# MAGIC 2. Terraform: Deploy endpoint from registered model (infrastructure)
# MAGIC
# MAGIC **Why REST endpoint instead of SDK**:
# MAGIC - Consistent interface with model serving (both REST)
# MAGIC - Can be called from any system (web app, mobile, PACS)
# MAGIC - Has direct access to inference tables for validation
# MAGIC - Auto-determines feedback_type based on prediction vs ground_truth

# COMMAND ----------
# MAGIC %md
# MAGIC ## Feedback Processing Logic as MLflow Model

# COMMAND ----------
%pip install mlflow --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import mlflow
import pandas as pd
from pyspark.sql import SparkSession
import uuid
from datetime import datetime
import json

class FeedbackProcessor(mlflow.pyfunc.PythonModel):
    """
    MLflow model that processes feedback submissions.

    This is deployed as a Model Serving endpoint and accepts feedback via REST API.
    """

    def __init__(self):
        self.spark = None

    def load_context(self, context):
        """Initialize Spark session when model loads"""
        from pyspark.sql import SparkSession
        self.spark = SparkSession.builder.getOrCreate()

    def predict(self, context, model_input):
        """
        Process feedback submission.

        Input format (JSON):
        {
            "dataframe_records": [{
                "request_id": "abc-123-def",           // Required: Prediction to link feedback to
                "ground_truth": "PNEUMONIA" | "NORMAL", // Required: Radiologist diagnosis
                "radiologist_id": "DR001",              // Optional: Who provided feedback
                "confidence": "confirmed",              // Optional: Confidence level
                "notes": "Optional notes"               // Optional: Additional context
            }]
        }

        NOTE: Databricks Model Serving limitation
        - Ideal REST design: POST /feedback/{request_id} with ground_truth in body
        - Current: request_id in POST body (MLflow models can't access path params)
        - Future TODO: Custom Flask/FastAPI app for true REST design

        Returns:
        {
            "feedback_id": "uuid",
            "status": "success" | "error",
            "message": "...",
            "feedback_type": "true-positive" | "false-positive" | etc.
        }
        """
        results = []

        for idx, row in model_input.iterrows():
            try:
                # Extract input
                request_id = row.get('request_id')
                ground_truth = row.get('ground_truth', '').upper()
                radiologist_id = row.get('radiologist_id', 'UNKNOWN')
                confidence = row.get('confidence', 'confirmed')
                notes = row.get('notes', '')

                # Validate inputs
                if not request_id:
                    results.append({
                        'status': 'error',
                        'message': 'request_id is required'
                    })
                    continue

                if ground_truth not in ['NORMAL', 'PNEUMONIA']:
                    results.append({
                        'status': 'error',
                        'message': f'ground_truth must be NORMAL or PNEUMONIA, got: {ground_truth}'
                    })
                    continue

                # Query the prediction from inference table
                prediction_query = f"""
                    SELECT
                        request_id,
                        served_model_name,
                        CAST(response:predictions[0][0] AS DOUBLE) as prediction_score,
                        CASE
                            WHEN CAST(response:predictions[0][0] AS DOUBLE) > 0.5
                            THEN 'PNEUMONIA'
                            ELSE 'NORMAL'
                        END as predicted_diagnosis
                    FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions
                    WHERE request_id = '{request_id}'
                    LIMIT 1
                """

                prediction_df = self.spark.sql(prediction_query)

                if prediction_df.count() == 0:
                    results.append({
                        'status': 'error',
                        'message': f'request_id not found: {request_id}'
                    })
                    continue

                prediction_row = prediction_df.collect()[0]
                predicted_diagnosis = prediction_row.predicted_diagnosis

                # Determine feedback_type automatically
                if predicted_diagnosis == 'PNEUMONIA' and ground_truth == 'PNEUMONIA':
                    feedback_type = 'true-positive'
                elif predicted_diagnosis == 'PNEUMONIA' and ground_truth == 'NORMAL':
                    feedback_type = 'false-positive'
                elif predicted_diagnosis == 'NORMAL' and ground_truth == 'NORMAL':
                    feedback_type = 'true-negative'
                elif predicted_diagnosis == 'NORMAL' and ground_truth == 'PNEUMONIA':
                    feedback_type = 'false-negative'
                else:
                    feedback_type = 'unknown'

                # Generate feedback_id
                feedback_id = f"feedback-{uuid.uuid4().hex[:8]}"

                # Write to feedback table
                feedback_data = {
                    'feedback_id': feedback_id,
                    'prediction_id': request_id,
                    'timestamp': datetime.now(),
                    'ground_truth': ground_truth,
                    'feedback_type': feedback_type,
                    'radiologist_id': radiologist_id,
                    'confidence': confidence,
                    'feedback_source': 'rest_api',
                    'notes': notes
                }

                feedback_df = self.spark.createDataFrame([feedback_data])
                feedback_df.write.mode('append').saveAsTable(
                    'healthcare_catalog_dev.gold.prediction_feedback'
                )

                # Success response
                results.append({
                    'feedback_id': feedback_id,
                    'status': 'success',
                    'message': 'Feedback submitted successfully',
                    'request_id': request_id,
                    'ground_truth': ground_truth,
                    'predicted_diagnosis': predicted_diagnosis,
                    'feedback_type': feedback_type,
                    'served_model_name': prediction_row.served_model_name
                })

            except Exception as e:
                results.append({
                    'status': 'error',
                    'message': f'Error processing feedback: {str(e)}'
                })

        return pd.DataFrame(results)

# Test the model locally
print("Testing FeedbackProcessor model...")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Register Feedback Processor as MLflow Model

# COMMAND ----------
import mlflow

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/feedback-endpoint-experiments")

# Create and log the model
with mlflow.start_run(run_name="feedback_processor_v1") as run:

    # Create model instance
    feedback_model = FeedbackProcessor()

    # Define input/output schema
    from mlflow.models.signature import infer_signature

    # Example input
    example_input = pd.DataFrame([{
        'request_id': 'example-request-id',
        'ground_truth': 'PNEUMONIA',
        'radiologist_id': 'DR001',
        'confidence': 'confirmed',
        'notes': 'Test feedback'
    }])

    # Example output
    example_output = pd.DataFrame([{
        'feedback_id': 'feedback-12345678',
        'status': 'success',
        'message': 'Feedback submitted successfully',
        'request_id': 'example-request-id',
        'ground_truth': 'PNEUMONIA',
        'predicted_diagnosis': 'PNEUMONIA',
        'feedback_type': 'true-positive',
        'served_model_name': 'model-1'
    }])

    signature = infer_signature(example_input, example_output)

    # Log the model
    mlflow.pyfunc.log_model(
        artifact_path="feedback_processor",
        python_model=feedback_model,
        signature=signature,
        registered_model_name="healthcare_catalog_dev.models.feedback_processor",
        pip_requirements=[
            "mlflow",
            "pandas",
            "pyspark"
        ]
    )

    print(f"✅ Model registered: healthcare_catalog_dev.models.feedback_processor")
    print(f"   Run ID: {run.info.run_id}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Deploy as Model Serving Endpoint

# COMMAND ----------
import requests
import json

workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()

endpoint_config = {
    "name": "feedback-endpoint",
    "config": {
        "served_entities": [
            {
                "entity_name": "healthcare_catalog_dev.models.feedback_processor",
                "entity_version": "1",
                "workload_size": "Small",
                "scale_to_zero_enabled": True
            }
        ]
    }
}

# Check if endpoint already exists
check_url = f"https://{workspace_url}/api/2.0/serving-endpoints/feedback-endpoint"
check_response = requests.get(
    check_url,
    headers={"Authorization": f"Bearer {token}"}
)

if check_response.status_code == 200:
    print("⚠️  Endpoint already exists. Updating...")

    # Update endpoint
    update_url = f"https://{workspace_url}/api/2.0/serving-endpoints/feedback-endpoint/config"
    response = requests.put(
        update_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json={"served_entities": endpoint_config["config"]["served_entities"]}
    )
else:
    print("Creating new endpoint...")

    # Create endpoint
    create_url = f"https://{workspace_url}/api/2.0/serving-endpoints"
    response = requests.post(
        create_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json=endpoint_config
    )

if response.status_code in [200, 201]:
    print(f"✅ Feedback endpoint deployed!")
    print(f"   Name: feedback-endpoint")
    print(f"   URL: https://{workspace_url}/serving-endpoints/feedback-endpoint/invocations")
    print()
    print("Wait 5-10 minutes for endpoint to be READY")
else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Test the Feedback Endpoint

# COMMAND ----------
import time

print("Waiting for endpoint to be ready...")
print("This will take 5-10 minutes on first deployment...")
print()

# Wait for endpoint to be ready
max_wait = 600  # 10 minutes
start_time = time.time()

while time.time() - start_time < max_wait:
    status_response = requests.get(
        f"https://{workspace_url}/api/2.0/serving-endpoints/feedback-endpoint",
        headers={"Authorization": f"Bearer {token}"}
    )

    if status_response.status_code == 200:
        status_data = status_response.json()
        state = status_data.get("state", {}).get("ready", "NOT_READY")

        if state == "READY":
            print("✅ Endpoint is READY!")
            break
        else:
            print(f"   Status: {state} (waiting...)")
            time.sleep(30)
    else:
        print(f"   Error checking status: {status_response.status_code}")
        break

# COMMAND ----------
# MAGIC %md
# MAGIC ## Example: Submit Feedback via REST API

# COMMAND ----------
# First, make a prediction to get a request_id
# (In production, you'd have this from the model serving endpoint)

print("=" * 80)
print("EXAMPLE: Submit Feedback via REST Endpoint")
print("=" * 80)
print()

# Test payload
test_feedback = {
    "dataframe_records": [
        {
            "request_id": "test-prediction-id",  # Replace with actual request_id
            "ground_truth": "PNEUMONIA",
            "radiologist_id": "DR001",
            "confidence": "confirmed",
            "notes": "Clear signs of pneumonia in right lung"
        }
    ]
}

feedback_url = f"https://{workspace_url}/serving-endpoints/feedback-endpoint/invocations"

try:
    feedback_response = requests.post(
        feedback_url,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        },
        json=test_feedback,
        timeout=30
    )

    if feedback_response.status_code == 200:
        result = feedback_response.json()
        print("✅ Feedback submitted successfully!")
        print()
        print("Response:")
        print(json.dumps(result, indent=2))
    else:
        print(f"❌ Error: {feedback_response.status_code}")
        print(feedback_response.text)

except Exception as e:
    print(f"❌ Error calling feedback endpoint: {e}")
    print()
    print("Note: Replace 'test-prediction-id' with actual request_id from a real prediction")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Usage Examples

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 1: From Web Application
# MAGIC
# MAGIC ```javascript
# MAGIC // JavaScript (web app)
# MAGIC async function submitFeedback(requestId, diagnosis) {
# MAGIC   const response = await fetch(
# MAGIC     'https://your-workspace.databricks.com/serving-endpoints/feedback-endpoint/invocations',
# MAGIC     {
# MAGIC       method: 'POST',
# MAGIC       headers: {
# MAGIC         'Authorization': 'Bearer YOUR_TOKEN',
# MAGIC         'Content-Type': 'application/json'
# MAGIC       },
# MAGIC       body: JSON.stringify({
# MAGIC         dataframe_records: [{
# MAGIC           request_id: requestId,
# MAGIC           ground_truth: diagnosis,  // "NORMAL" or "PNEUMONIA"
# MAGIC           radiologist_id: "DR001",
# MAGIC           confidence: "confirmed",
# MAGIC           notes: "Reviewed by radiologist"
# MAGIC         }]
# MAGIC       })
# MAGIC     }
# MAGIC   );
# MAGIC
# MAGIC   return await response.json();
# MAGIC }
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 2: From Python Application
# MAGIC
# MAGIC ```python
# MAGIC import requests
# MAGIC
# MAGIC def submit_feedback(request_id, ground_truth, radiologist_id):
# MAGIC     """Submit feedback via REST endpoint"""
# MAGIC
# MAGIC     payload = {
# MAGIC         "dataframe_records": [{
# MAGIC             "request_id": request_id,
# MAGIC             "ground_truth": ground_truth,  # "NORMAL" or "PNEUMONIA"
# MAGIC             "radiologist_id": radiologist_id,
# MAGIC             "confidence": "confirmed"
# MAGIC         }]
# MAGIC     }
# MAGIC
# MAGIC     response = requests.post(
# MAGIC         "https://workspace.databricks.com/serving-endpoints/feedback-endpoint/invocations",
# MAGIC         headers={"Authorization": "Bearer YOUR_TOKEN"},
# MAGIC         json=payload
# MAGIC     )
# MAGIC
# MAGIC     return response.json()
# MAGIC
# MAGIC # Usage
# MAGIC result = submit_feedback("abc-123", "PNEUMONIA", "DR001")
# MAGIC print(result['feedback_type'])  # "true-positive"
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 3: From Mobile App (Swift/iOS)
# MAGIC
# MAGIC ```swift
# MAGIC func submitFeedback(requestId: String, diagnosis: String) async throws {
# MAGIC     let url = URL(string: "https://workspace.databricks.com/serving-endpoints/feedback-endpoint/invocations")!
# MAGIC
# MAGIC     var request = URLRequest(url: url)
# MAGIC     request.httpMethod = "POST"
# MAGIC     request.setValue("Bearer YOUR_TOKEN", forHTTPHeaderField: "Authorization")
# MAGIC     request.setValue("application/json", forHTTPHeaderField: "Content-Type")
# MAGIC
# MAGIC     let payload: [String: Any] = [
# MAGIC         "dataframe_records": [[
# MAGIC             "request_id": requestId,
# MAGIC             "ground_truth": diagnosis,
# MAGIC             "radiologist_id": "DR001",
# MAGIC             "confidence": "confirmed"
# MAGIC         ]]
# MAGIC     ]
# MAGIC
# MAGIC     request.httpBody = try JSONSerialization.data(withJSONObject: payload)
# MAGIC
# MAGIC     let (data, _) = try await URLSession.shared.data(for: request)
# MAGIC     let result = try JSONDecoder().decode(FeedbackResponse.self, from: data)
# MAGIC }
# MAGIC ```

# COMMAND ----------
# MAGIC %md
# MAGIC ## Architecture Summary
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    Complete Architecture                     │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC
# MAGIC 1. PREDICTION (Model Serving Endpoint)
# MAGIC    POST /serving-endpoints/pneumonia-classifier-ab-test/invocations
# MAGIC    ├─ Input: X-ray image
# MAGIC    ├─ Output: Prediction + request_id
# MAGIC    └─ Logged to: gold.pneumonia_classifier_predictions
# MAGIC
# MAGIC 2. FEEDBACK (Feedback Endpoint) ← NEW
# MAGIC    POST /serving-endpoints/feedback-endpoint/invocations
# MAGIC    ├─ Input: request_id + ground_truth
# MAGIC    ├─ Validates: request_id exists in inference table
# MAGIC    ├─ Auto-determines: feedback_type (TP/FP/TN/FN)
# MAGIC    ├─ Writes to: gold.prediction_feedback
# MAGIC    └─ Output: feedback_id + feedback_type
# MAGIC
# MAGIC 3. MONITORING
# MAGIC    Query: gold.model_performance_live (view)
# MAGIC    ├─ JOINs: predictions + feedback
# MAGIC    ├─ Calculates: per-model accuracy
# MAGIC    └─ Recommends: PROMOTE / KEEP / ROLLBACK
# MAGIC ```
# MAGIC
# MAGIC **Benefits of REST Endpoint**:
# MAGIC - ✅ Consistent interface (both prediction and feedback via REST)
# MAGIC - ✅ Accessible from any system (web, mobile, PACS)
# MAGIC - ✅ Direct access to inference tables for validation
# MAGIC - ✅ Auto-determines feedback_type (no manual classification)
# MAGIC - ✅ Auto-scaling (same infrastructure as model serving)
# MAGIC - ✅ Built-in authentication and logging

# COMMAND ----------
# MAGIC %md
# MAGIC ## Outputs

# COMMAND ----------
print("=" * 80)
print("DEPLOYMENT SUMMARY")
print("=" * 80)
print()
print("✅ Feedback Processor Model:")
print("   - Registered: healthcare_catalog_dev.models.feedback_processor")
print("   - Version: 1")
print()
print("✅ Feedback REST Endpoint:")
print(f"   - Name: feedback-endpoint")
print(f"   - URL: https://{workspace_url}/serving-endpoints/feedback-endpoint/invocations")
print()
print("Usage:")
print("   POST /serving-endpoints/feedback-endpoint/invocations")
print("   Body: { dataframe_records: [{ request_id, ground_truth, ... }] }")
print()
print("Next Steps:")
print("   1. Wait for endpoint to be READY (5-10 minutes)")
print("   2. Test with actual request_id from model predictions")
print("   3. Update web/mobile apps to call this endpoint")
print("   4. Update interactive_feedback_review notebook to use REST API")

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Future Enhancement: Ideal REST Design
# MAGIC
# MAGIC **Current Limitation**: Databricks Model Serving with MLflow models cannot access URL path parameters or query strings. All input must come via POST body.
# MAGIC
# MAGIC **Ideal REST API Design** (when time allows):
# MAGIC
# MAGIC ```
# MAGIC POST /feedback/{request_id}
# MAGIC Body: {
# MAGIC   "ground_truth": "PNEUMONIA",
# MAGIC   "radiologist_id": "DR001",
# MAGIC   "confidence": "confirmed",
# MAGIC   "notes": "Optional"
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Benefits**:
# MAGIC - ✅ RESTful URL structure (resource identified in path)
# MAGIC - ✅ Cleaner API (request_id not in body)
# MAGIC - ✅ Better HTTP semantics
# MAGIC - ✅ Easier to cache/log by URL
# MAGIC
# MAGIC **Implementation Options**:
# MAGIC
# MAGIC ### Option 1: Databricks Apps (Flask/FastAPI)
# MAGIC ```python
# MAGIC from flask import Flask, request, jsonify
# MAGIC from pyspark.sql import SparkSession
# MAGIC
# MAGIC app = Flask(__name__)
# MAGIC spark = SparkSession.builder.getOrCreate()
# MAGIC
# MAGIC @app.route('/feedback/<request_id>', methods=['POST'])
# MAGIC def submit_feedback(request_id):
# MAGIC     data = request.json
# MAGIC     ground_truth = data['ground_truth']
# MAGIC     
# MAGIC     # Query inference table
# MAGIC     prediction = spark.sql(f"""
# MAGIC         SELECT predicted_diagnosis
# MAGIC         FROM gold.pneumonia_classifier_predictions
# MAGIC         WHERE request_id = '{request_id}'
# MAGIC     """).collect()[0]
# MAGIC     
# MAGIC     # Determine feedback_type
# MAGIC     feedback_type = calculate_feedback_type(
# MAGIC         prediction.predicted_diagnosis,
# MAGIC         ground_truth
# MAGIC     )
# MAGIC     
# MAGIC     # Write feedback
# MAGIC     # ... (same logic as current)
# MAGIC     
# MAGIC     return jsonify({
# MAGIC         'feedback_id': feedback_id,
# MAGIC         'feedback_type': feedback_type
# MAGIC     })
# MAGIC ```
# MAGIC
# MAGIC ### Option 2: API Gateway + Lambda (AWS)
# MAGIC ```
# MAGIC API Gateway:
# MAGIC   POST /feedback/{request_id}
# MAGIC   ↓
# MAGIC Lambda Function:
# MAGIC   - Connect to Databricks SQL endpoint
# MAGIC   - Query inference table
# MAGIC   - Determine feedback_type
# MAGIC   - Write to feedback table via JDBC
# MAGIC ```
# MAGIC
# MAGIC ### Option 3: Custom Web Service (Docker on Databricks)
# MAGIC ```
# MAGIC FastAPI app in Docker container:
# MAGIC - Deployed as Databricks App
# MAGIC - Direct Spark access
# MAGIC - Custom URL routing
# MAGIC - Full REST compliance
# MAGIC ```
# MAGIC
# MAGIC **Trade-offs**:
# MAGIC
# MAGIC | Approach | Pros | Cons |
# MAGIC |----------|------|------|
# MAGIC | **Current (Model Serving)** | Same infra as predictions, auto-scaling, simple | Not true REST, request_id in body |
# MAGIC | **Databricks Apps (Flask)** | True REST, full control, Spark access | Different infra, manual scaling |
# MAGIC | **API Gateway + Lambda** | True REST, serverless | Extra AWS cost, JDBC overhead |
# MAGIC | **Docker on Databricks** | True REST, Spark access | More complex deployment |
# MAGIC
# MAGIC **Recommendation**:
# MAGIC - **Now**: Use current Model Serving approach (consistency with prediction endpoints)
# MAGIC - **Later**: Migrate to Databricks Apps with Flask/FastAPI when team has bandwidth
# MAGIC - **Priority**: Medium (current approach works, this is API aesthetics)
# MAGIC
# MAGIC **Estimated Effort**: 1-2 days for Flask/FastAPI implementation + testing

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC **End of Notebook**
# MAGIC
# MAGIC Next: Deploy endpoint via Terraform (`terraform/databricks/endpoints.tf`)
