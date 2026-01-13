# Table Schemas - Source of Truth

**All tables defined in `terraform/databricks/tables.tf`**

These are the ONLY tables we use. Ignore any auto-created tables from Databricks.

---

## Gold Layer Tables (Business-Ready ML Outputs)

### 1. `healthcare_catalog_dev.gold.pneumonia_predictions`

**Purpose**: ML model predictions for pneumonia classification

**Schema**:
```sql
prediction_id         STRING      -- Unique identifier for prediction
image_id              STRING      -- Foreign key to silver.xray_metadata
predicted_label       INT         -- Predicted label: 0=NORMAL, 1=PNEUMONIA
prediction_probability DOUBLE     -- Probability of pneumonia (0-1)
confidence_score      DOUBLE      -- Model confidence in prediction (0-1)
true_label            INT         -- Actual label for validation
is_correct            BOOLEAN     -- Whether prediction matches true label
model_name            STRING      -- Name of ML model used
model_version         STRING      -- Version of ML model
predicted_at          TIMESTAMP   -- Timestamp when prediction was made
prediction_date       DATE        -- Date partition for predictions
```

**Usage**:
- Store ALL predictions from model serving endpoints
- Link predictions to images via `image_id`
- Track which model version made each prediction
- Support model performance analysis

---

### 2. `healthcare_catalog_dev.gold.prediction_feedback`

**Purpose**: Ground truth labels from radiologists for model predictions

**Schema**:
```sql
feedback_id       STRING      NOT NULL -- Unique feedback identifier (UUID)
prediction_id     STRING      NOT NULL -- Links to prediction_id from pneumonia_predictions
timestamp         TIMESTAMP   NOT NULL -- When feedback was submitted
ground_truth      STRING      NOT NULL -- Actual diagnosis: NORMAL or PNEUMONIA
feedback_type     STRING               -- Classification: true-positive, false-positive, true-negative, false-negative
radiologist_id    STRING               -- ID of radiologist providing feedback
confidence        STRING               -- Confidence level: confirmed, uncertain, needs_review
feedback_source   STRING               -- How feedback was collected: api, radiologist, pathology, manual
notes             STRING               -- Optional notes from radiologist
```

**Usage**:
- Link feedback to predictions via `prediction_id`
- Store ground truth from radiologists
- Auto-calculate `feedback_type` (TP/FP/TN/FN)
- Enable accuracy tracking per model

---

### 3. `healthcare_catalog_dev.gold.model_performance`

**Purpose**: Model performance metrics and evaluation results

**Schema**:
```sql
evaluation_id     STRING      -- Unique identifier for evaluation run
model_name        STRING      -- Name of ML model
model_version     STRING      -- Version of ML model
dataset_split     STRING      -- Dataset used: train, test, or val
accuracy          DOUBLE      -- Overall accuracy
precision         DOUBLE      -- Precision score
recall            DOUBLE      -- Recall score
f1_score          DOUBLE      -- F1 score
auc_roc           DOUBLE      -- Area under ROC curve
confusion_matrix  STRING      -- Confusion matrix as JSON
total_samples     BIGINT      -- Total number of samples evaluated
evaluated_at      TIMESTAMP   -- Timestamp when evaluation was run
evaluation_date   DATE        -- Date partition for evaluations
```

**Usage**:
- Store aggregated performance metrics
- Compare model versions
- Track performance over time
- Support model promotion decisions

---

## Silver Layer Tables (Cleaned Data)

### 4. `healthcare_catalog_dev.silver.xray_metadata`

**Schema**:
```sql
image_id          STRING      -- Unique identifier for X-ray image
label             INT         -- Binary label: 0=NORMAL, 1=PNEUMONIA
label_name        STRING      -- Human-readable label
dataset_split     STRING      -- train, test, or val split
file_path         STRING      -- Validated path to image file
image_width       INT         -- Image width in pixels
image_height      INT         -- Image height in pixels
quality_score     DOUBLE      -- Image quality score (0-1)
processed_at      TIMESTAMP   -- Timestamp when data was processed
source_batch_id   STRING      -- Reference to bronze batch ID
```

---

## Bronze Layer Tables (Raw Data)

### 5. `healthcare_catalog_dev.bronze.kaggle_xray_metadata`

**Schema**:
```sql
image_id          STRING      -- Unique identifier for X-ray image
filename          STRING      -- Original filename from Kaggle
category          STRING      -- Image category: NORMAL or PNEUMONIA
dataset_split     STRING      -- train, test, or val split
file_path         STRING      -- Path to image file in volume
file_size_bytes   BIGINT      -- File size in bytes
ingested_at       TIMESTAMP   -- Timestamp when data was ingested
ingestion_batch_id STRING     -- Batch ID for tracking ingestion runs
```

---

## Key Relationships

```
bronze.kaggle_xray_metadata
    ↓ (ETL)
silver.xray_metadata
    ↓ (ML Inference)
gold.pneumonia_predictions ←→ gold.prediction_feedback
    ↓ (Aggregation)          (via prediction_id)
gold.model_performance
```

---

## IMPORTANT: Auto-Capture Tables (IGNORE THESE)

Databricks Model Serving with `auto_capture_config` enabled creates tables automatically:
- `pneumonia_classifier_payload` (request inputs)
- `pneumonia_classifier_predictions` (raw responses)

**DO NOT USE THESE TABLES.** They have a different schema and are not controlled by our Terraform.

**Use our Terraform-defined tables instead:**
- `gold.pneumonia_predictions` (our controlled schema)
- `gold.prediction_feedback` (our controlled schema)

---

## Migration Notes

If you need to migrate data from auto-capture tables to our tables:

```sql
-- Example: Copy from auto-capture to our schema
INSERT INTO healthcare_catalog_dev.gold.pneumonia_predictions
SELECT
    request_id as prediction_id,
    NULL as image_id,  -- Need to link manually
    CASE WHEN CAST(response:predictions[0][0] AS DOUBLE) > 0.5 THEN 1 ELSE 0 END as predicted_label,
    CAST(response:predictions[0][0] AS DOUBLE) as prediction_probability,
    CAST(response:predictions[0][0] AS DOUBLE) as confidence_score,
    NULL as true_label,
    NULL as is_correct,
    served_model_name as model_name,
    SPLIT(served_model_name, '-')[1] as model_version,
    FROM_UNIXTIME(timestamp_ms / 1000) as predicted_at,
    DATE(FROM_UNIXTIME(timestamp_ms / 1000)) as prediction_date
FROM pneumonia_classifier_predictions;  -- Auto-capture table
```

But ideally: **Write to our tables directly from code, ignore auto-capture.**
