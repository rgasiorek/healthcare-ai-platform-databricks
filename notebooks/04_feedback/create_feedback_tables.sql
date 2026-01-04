-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Feedback Infrastructure - Table Creation
-- MAGIC
-- MAGIC This notebook creates the feedback loop infrastructure for tracking model performance:
-- MAGIC 1. **prediction_feedback** table - stores ground truth labels from radiologists
-- MAGIC 2. **model_performance_live** view - joins predictions with feedback for analysis
-- MAGIC
-- MAGIC **Educational Note**:
-- MAGIC - Feedback tables enable the "closed loop" in MLOps
-- MAGIC - Links predictions (model output) with ground truth (doctor diagnosis)
-- MAGIC - Enables Champion/Challenger comparison and model improvement
-- MAGIC
-- MAGIC **Prerequisites**: A/B testing endpoint must be deployed with inference logging enabled

-- COMMAND ----------
-- MAGIC %md
-- MAGIC ## Step 1: Create Prediction Feedback Table

-- COMMAND ----------
-- Drop table if exists (for development/testing)
-- DROP TABLE IF EXISTS healthcare_catalog_dev.gold.prediction_feedback;

CREATE TABLE IF NOT EXISTS healthcare_catalog_dev.gold.prediction_feedback (
    feedback_id STRING NOT NULL COMMENT 'Unique feedback identifier (UUID)',
    prediction_id STRING NOT NULL COMMENT 'Links to Databricks request_id from inference table',
    timestamp TIMESTAMP NOT NULL COMMENT 'When feedback was submitted',
    ground_truth STRING NOT NULL COMMENT 'Actual diagnosis: NORMAL or PNEUMONIA',
    feedback_type STRING COMMENT 'Classification: true-positive, false-positive, true-negative, false-negative',
    radiologist_id STRING COMMENT 'ID of radiologist providing feedback',
    confidence STRING COMMENT 'Confidence level: confirmed, uncertain, needs_review',
    feedback_source STRING COMMENT 'How feedback was collected: api, radiologist, pathology, manual',
    notes STRING COMMENT 'Optional notes from radiologist'
)
USING DELTA
COMMENT 'Ground truth labels for model predictions - enables accuracy tracking and A/B testing'
PARTITIONED BY (DATE(timestamp))
TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true',
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- COMMAND ----------
-- MAGIC %md
-- MAGIC ## Step 2: Verify Table Creation

-- COMMAND ----------
DESCRIBE TABLE EXTENDED healthcare_catalog_dev.gold.prediction_feedback;

-- COMMAND ----------
-- Show sample structure (empty initially)
SELECT * FROM healthcare_catalog_dev.gold.prediction_feedback LIMIT 5;

-- COMMAND ----------
-- MAGIC %md
-- MAGIC ## Step 3: Create Model Performance Analysis View
-- MAGIC
-- MAGIC This view joins inference predictions with feedback to calculate per-model accuracy

-- COMMAND ----------
CREATE OR REPLACE VIEW healthcare_catalog_dev.gold.model_performance_live AS
SELECT
    -- Prediction info
    p.request_id,
    p.date as prediction_date,
    p.timestamp_ms as prediction_timestamp,
    p.served_model_name,
    p.status_code,

    -- Extract prediction from JSON response
    CAST(p.response:predictions[0][0] AS DOUBLE) as prediction_score,
    CASE
        WHEN CAST(p.response:predictions[0][0] AS DOUBLE) > 0.5
        THEN 'PNEUMONIA'
        ELSE 'NORMAL'
    END as predicted_class,

    -- Feedback info (NULL if no feedback yet)
    f.feedback_id,
    f.ground_truth,
    f.feedback_type,
    f.radiologist_id,
    f.confidence as feedback_confidence,
    f.timestamp as feedback_timestamp,

    -- Correctness flag
    CASE
        WHEN f.ground_truth IS NULL THEN NULL  -- No feedback yet
        WHEN (CAST(p.response:predictions[0][0] AS DOUBLE) > 0.5 AND f.ground_truth = 'PNEUMONIA')
            OR (CAST(p.response:predictions[0][0] AS DOUBLE) <= 0.5 AND f.ground_truth = 'NORMAL')
        THEN TRUE
        ELSE FALSE
    END as is_correct,

    -- Confusion matrix classification
    CASE
        WHEN f.ground_truth IS NULL THEN 'NO_FEEDBACK'
        WHEN CAST(p.response:predictions[0][0] AS DOUBLE) > 0.5 AND f.ground_truth = 'PNEUMONIA' THEN 'TRUE_POSITIVE'
        WHEN CAST(p.response:predictions[0][0] AS DOUBLE) > 0.5 AND f.ground_truth = 'NORMAL' THEN 'FALSE_POSITIVE'
        WHEN CAST(p.response:predictions[0][0] AS DOUBLE) <= 0.5 AND f.ground_truth = 'NORMAL' THEN 'TRUE_NEGATIVE'
        WHEN CAST(p.response:predictions[0][0] AS DOUBLE) <= 0.5 AND f.ground_truth = 'PNEUMONIA' THEN 'FALSE_NEGATIVE'
    END as confusion_label

FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
    ON p.request_id = f.prediction_id

COMMENT 'Live view of model predictions joined with feedback for performance analysis';

-- COMMAND ----------
-- MAGIC %md
-- MAGIC ## Step 4: Create Helper View - Model Comparison Summary

-- COMMAND ----------
CREATE OR REPLACE VIEW healthcare_catalog_dev.gold.model_comparison_summary AS
SELECT
    served_model_name,
    date(prediction_date) as date,

    -- Prediction counts
    COUNT(*) as total_predictions,
    COUNT(feedback_id) as feedback_count,
    ROUND(COUNT(feedback_id) * 100.0 / COUNT(*), 2) as feedback_coverage_pct,

    -- Accuracy metrics (only for predictions with feedback)
    SUM(CASE WHEN is_correct = TRUE THEN 1 ELSE 0 END) as correct_predictions,
    SUM(CASE WHEN is_correct = FALSE THEN 1 ELSE 0 END) as incorrect_predictions,
    ROUND(AVG(CASE WHEN is_correct = TRUE THEN 1.0 WHEN is_correct = FALSE THEN 0.0 END) * 100, 2) as accuracy_pct,

    -- Confusion matrix
    SUM(CASE WHEN confusion_label = 'TRUE_POSITIVE' THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN confusion_label = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN confusion_label = 'TRUE_NEGATIVE' THEN 1 ELSE 0 END) as true_negatives,
    SUM(CASE WHEN confusion_label = 'FALSE_NEGATIVE' THEN 1 ELSE 0 END) as false_negatives,

    -- Precision and Recall
    ROUND(
        SUM(CASE WHEN confusion_label = 'TRUE_POSITIVE' THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(CASE WHEN confusion_label IN ('TRUE_POSITIVE', 'FALSE_POSITIVE') THEN 1 ELSE 0 END), 0),
        2
    ) as precision_pct,
    ROUND(
        SUM(CASE WHEN confusion_label = 'TRUE_POSITIVE' THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(CASE WHEN confusion_label IN ('TRUE_POSITIVE', 'FALSE_NEGATIVE') THEN 1 ELSE 0 END), 0),
        2
    ) as recall_pct,

    -- Average prediction confidence
    ROUND(AVG(prediction_score), 4) as avg_prediction_score

FROM healthcare_catalog_dev.gold.model_performance_live
GROUP BY served_model_name, date(prediction_date)
ORDER BY date DESC, served_model_name

COMMENT 'Daily summary of model performance - Champion vs Challenger comparison';

-- COMMAND ----------
-- MAGIC %md
-- MAGIC ## Step 5: Test Queries

-- COMMAND ----------
-- Show view structure
DESCRIBE healthcare_catalog_dev.gold.model_performance_live;

-- COMMAND ----------
-- Sample query: Recent predictions (will be empty until A/B endpoint is deployed)
SELECT
    prediction_timestamp,
    served_model_name,
    predicted_class,
    prediction_score,
    ground_truth,
    is_correct
FROM healthcare_catalog_dev.gold.model_performance_live
ORDER BY prediction_timestamp DESC
LIMIT 10;

-- COMMAND ----------
-- MAGIC %md
-- MAGIC ## Summary
-- MAGIC
-- MAGIC ### Tables Created:
-- MAGIC 1. ✅ `healthcare_catalog_dev.gold.prediction_feedback` - stores ground truth labels
-- MAGIC 2. ✅ `healthcare_catalog_dev.gold.model_performance_live` - joins predictions + feedback
-- MAGIC 3. ✅ `healthcare_catalog_dev.gold.model_comparison_summary` - daily model comparison
-- MAGIC
-- MAGIC ### Next Steps:
-- MAGIC 1. Deploy A/B testing endpoint (will populate inference tables)
-- MAGIC 2. Create feedback collector to submit ground truth labels
-- MAGIC 3. Use monitoring dashboard to compare Champion vs Challenger
-- MAGIC
-- MAGIC ### Educational Note:
-- MAGIC This infrastructure enables the **feedback loop** in MLOps:
-- MAGIC - Models make predictions → captured in inference tables
-- MAGIC - Doctors provide labels → stored in feedback table
-- MAGIC - System joins both → calculates accuracy per model
-- MAGIC - Data-driven decision → promote best model to Champion
-- MAGIC
-- MAGIC **Key insight**: Without feedback, we can't measure real-world performance!
