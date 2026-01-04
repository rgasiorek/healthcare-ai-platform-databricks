# Databricks notebook source
# MAGIC %md
# MAGIC # Feedback Collector - BentoML-Style API
# MAGIC
# MAGIC Simple API for submitting feedback on model predictions.
# MAGIC Enables radiologists to mark predictions as correct/incorrect.
# MAGIC
# MAGIC **Design Philosophy**: Keep it simple, like BentoML
# MAGIC - URL-style interface: `/feedback/{prediction_id}/{feedback_type}`
# MAGIC - Automatic ground truth mapping from feedback type
# MAGIC - Direct Delta table writes (no external services needed)
# MAGIC
# MAGIC **Usage**:
# MAGIC ```python
# MAGIC from feedback_collector import submit_feedback
# MAGIC
# MAGIC feedback_id = submit_feedback(
# MAGIC     prediction_id="abc-123",
# MAGIC     feedback_type="true-positive",
# MAGIC     radiologist_id="DR001"
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC **Prerequisites**: Feedback table must exist (run `create_feedback_tables.sql` first)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Imports and Setup

# COMMAND ----------
import uuid
from datetime import datetime
from pyspark.sql import SparkSession
from typing import Optional, Dict, List

# COMMAND ----------
# MAGIC %md
# MAGIC ## Core Feedback Functions

# COMMAND ----------
def submit_feedback(
    prediction_id: str,
    feedback_type: str,
    radiologist_id: Optional[str] = None,
    confidence: str = "confirmed",
    feedback_source: str = "manual",
    notes: Optional[str] = None
) -> str:
    """
    Submit feedback for a model prediction (BentoML-style)

    Args:
        prediction_id: The Databricks request_id from model serving
        feedback_type: One of:
            - 'true-positive': Model predicted PNEUMONIA, correct
            - 'false-positive': Model predicted PNEUMONIA, wrong (was NORMAL)
            - 'true-negative': Model predicted NORMAL, correct
            - 'false-negative': Model predicted NORMAL, wrong (was PNEUMONIA)
        radiologist_id: Optional doctor ID (e.g., "DR001")
        confidence: Confidence level ('confirmed', 'uncertain', 'needs_review')
        feedback_source: How feedback was collected ('manual', 'api', 'radiologist', 'pathology')
        notes: Optional notes from radiologist

    Returns:
        feedback_id: UUID of created feedback record

    Example:
        >>> feedback_id = submit_feedback("abc-123", "true-positive", "DR001")
        ✅ Feedback submitted: feedback-uuid-456
           Prediction ID: abc-123
           Ground Truth: PNEUMONIA
    """

    # Map feedback_type to ground_truth
    feedback_mapping = {
        "true-positive": {"ground_truth": "PNEUMONIA", "predicted": "PNEUMONIA", "correct": True},
        "false-positive": {"ground_truth": "NORMAL", "predicted": "PNEUMONIA", "correct": False},
        "true-negative": {"ground_truth": "NORMAL", "predicted": "NORMAL", "correct": True},
        "false-negative": {"ground_truth": "PNEUMONIA", "predicted": "NORMAL", "correct": False}
    }

    # Validate feedback_type
    if feedback_type not in feedback_mapping:
        raise ValueError(
            f"Invalid feedback_type: '{feedback_type}'. "
            f"Must be one of {list(feedback_mapping.keys())}"
        )

    # Generate unique feedback ID
    feedback_id = str(uuid.uuid4())
    ground_truth = feedback_mapping[feedback_type]["ground_truth"]

    # Create feedback record
    feedback_record = {
        "feedback_id": feedback_id,
        "prediction_id": prediction_id,
        "timestamp": datetime.now(),
        "ground_truth": ground_truth,
        "feedback_type": feedback_type,
        "radiologist_id": radiologist_id,
        "confidence": confidence,
        "feedback_source": feedback_source,
        "notes": notes
    }

    # Write to Delta table
    feedback_df = spark.createDataFrame([feedback_record])
    feedback_df.write.mode("append").saveAsTable(
        "healthcare_catalog_dev.gold.prediction_feedback"
    )

    # Print confirmation
    print(f"✅ Feedback submitted: {feedback_id}")
    print(f"   Prediction ID: {prediction_id}")
    print(f"   Ground Truth: {ground_truth}")
    print(f"   Feedback Type: {feedback_type}")
    if radiologist_id:
        print(f"   Radiologist: {radiologist_id}")

    return feedback_id


# COMMAND ----------
def submit_feedback_batch(feedback_records: List[Dict]) -> List[str]:
    """
    Submit multiple feedbacks at once (batch operation)

    Args:
        feedback_records: List of dicts, each containing:
            - prediction_id (required)
            - feedback_type (required)
            - radiologist_id (optional)
            - confidence (optional)
            - notes (optional)

    Returns:
        List of feedback_ids

    Example:
        >>> records = [
        ...     {"prediction_id": "abc-123", "feedback_type": "true-positive"},
        ...     {"prediction_id": "def-456", "feedback_type": "false-positive"}
        ... ]
        >>> feedback_ids = submit_feedback_batch(records)
        ✅ Batch feedback submitted: 2 records
    """

    feedback_mapping = {
        "true-positive": "PNEUMONIA",
        "false-positive": "NORMAL",
        "true-negative": "NORMAL",
        "false-negative": "PNEUMONIA"
    }

    processed_records = []
    feedback_ids = []

    for record in feedback_records:
        # Validate required fields
        if "prediction_id" not in record or "feedback_type" not in record:
            raise ValueError("Each record must have 'prediction_id' and 'feedback_type'")

        feedback_type = record["feedback_type"]
        if feedback_type not in feedback_mapping:
            raise ValueError(f"Invalid feedback_type: '{feedback_type}'")

        # Generate feedback ID and ground truth
        feedback_id = str(uuid.uuid4())
        feedback_ids.append(feedback_id)

        processed_record = {
            "feedback_id": feedback_id,
            "prediction_id": record["prediction_id"],
            "timestamp": datetime.now(),
            "ground_truth": feedback_mapping[feedback_type],
            "feedback_type": feedback_type,
            "radiologist_id": record.get("radiologist_id"),
            "confidence": record.get("confidence", "confirmed"),
            "feedback_source": record.get("feedback_source", "manual"),
            "notes": record.get("notes")
        }

        processed_records.append(processed_record)

    # Write batch to Delta table
    feedback_df = spark.createDataFrame(processed_records)
    feedback_df.write.mode("append").saveAsTable(
        "healthcare_catalog_dev.gold.prediction_feedback"
    )

    print(f"✅ Batch feedback submitted: {len(processed_records)} records")

    return feedback_ids


# COMMAND ----------
def get_prediction_info(prediction_id: str) -> Optional[Dict]:
    """
    Get information about a prediction (what was predicted)

    Args:
        prediction_id: The Databricks request_id

    Returns:
        Dict with prediction info, or None if not found

    Example:
        >>> info = get_prediction_info("abc-123")
        >>> print(info)
        {
            'prediction_id': 'abc-123',
            'served_model_name': 'pneumonia_classifier_keras-1',
            'predicted_class': 'PNEUMONIA',
            'prediction_score': 0.85,
            'timestamp': '2025-01-04 10:30:00'
        }
    """

    try:
        result = spark.sql(f"""
            SELECT
                request_id as prediction_id,
                served_model_name,
                timestamp_ms as timestamp,
                CAST(response:predictions[0][0] AS DOUBLE) as prediction_score,
                CASE
                    WHEN CAST(response:predictions[0][0] AS DOUBLE) > 0.5
                    THEN 'PNEUMONIA'
                    ELSE 'NORMAL'
                END as predicted_class
            FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions
            WHERE request_id = '{prediction_id}'
            LIMIT 1
        """).collect()

        if result:
            row = result[0]
            return {
                "prediction_id": row.prediction_id,
                "served_model_name": row.served_model_name,
                "predicted_class": row.predicted_class,
                "prediction_score": float(row.prediction_score),
                "timestamp": str(row.timestamp)
            }
        else:
            print(f"⚠️  Prediction not found: {prediction_id}")
            return None

    except Exception as e:
        print(f"❌ Error fetching prediction: {e}")
        return None


# COMMAND ----------
def get_feedback_stats(days: int = 7) -> Dict:
    """
    Get feedback collection statistics

    Args:
        days: Number of days to look back (default: 7)

    Returns:
        Dict with feedback stats

    Example:
        >>> stats = get_feedback_stats(days=7)
        >>> print(stats)
        {
            'total_predictions': 2100,
            'feedback_count': 941,
            'feedback_coverage_pct': 44.8,
            'by_model': {...}
        }
    """

    try:
        result = spark.sql(f"""
            SELECT
                COUNT(DISTINCT p.request_id) as total_predictions,
                COUNT(DISTINCT f.feedback_id) as feedback_count,
                ROUND(COUNT(DISTINCT f.feedback_id) * 100.0 / COUNT(DISTINCT p.request_id), 2) as feedback_coverage_pct,

                -- By model
                COLLECT_LIST(
                    STRUCT(
                        p.served_model_name,
                        COUNT(DISTINCT CASE WHEN f.feedback_id IS NOT NULL THEN p.request_id END) as feedback_count
                    )
                ) as by_model

            FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
            LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
                ON p.request_id = f.prediction_id
            WHERE p.date >= current_date() - INTERVAL {days} DAYS
        """).collect()

        if result:
            row = result[0]
            return {
                "total_predictions": int(row.total_predictions),
                "feedback_count": int(row.feedback_count),
                "feedback_coverage_pct": float(row.feedback_coverage_pct),
                "days": days
            }
        else:
            return {
                "total_predictions": 0,
                "feedback_count": 0,
                "feedback_coverage_pct": 0.0,
                "days": days
            }

    except Exception as e:
        print(f"❌ Error fetching stats: {e}")
        return {}


# COMMAND ----------
# MAGIC %md
# MAGIC ## Example Usage

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 1: Submit Single Feedback

# COMMAND ----------
# Example prediction ID (will vary in real use)
# prediction_id = "abc-def-123-456"  # From model serving response

# Submit feedback
# feedback_id = submit_feedback(
#     prediction_id=prediction_id,
#     feedback_type="true-positive",
#     radiologist_id="DR001",
#     notes="Clear consolidation in right lower lobe"
# )

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 2: Get Prediction Info Before Submitting Feedback

# COMMAND ----------
# Check what the model predicted
# info = get_prediction_info(prediction_id)
# if info:
#     print(f"Model: {info['served_model_name']}")
#     print(f"Predicted: {info['predicted_class']} ({info['prediction_score']:.2f})")
#     print(f"Time: {info['timestamp']}")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 3: Batch Feedback Submission

# COMMAND ----------
# feedback_records = [
#     {"prediction_id": "abc-123", "feedback_type": "true-positive", "radiologist_id": "DR001"},
#     {"prediction_id": "def-456", "feedback_type": "false-positive", "radiologist_id": "DR001"},
#     {"prediction_id": "ghi-789", "feedback_type": "true-negative", "radiologist_id": "DR002"}
# ]

# feedback_ids = submit_feedback_batch(feedback_records)
# print(f"Submitted {len(feedback_ids)} feedbacks")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Example 4: Check Feedback Coverage

# COMMAND ----------
# stats = get_feedback_stats(days=7)
# print(f"Total predictions: {stats['total_predictions']}")
# print(f"Feedback collected: {stats['feedback_count']}")
# print(f"Coverage: {stats['feedback_coverage_pct']:.1f}%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC ### Functions Available:
# MAGIC
# MAGIC 1. **`submit_feedback()`** - Submit single feedback (BentoML-style)
# MAGIC    - Simple URL-like interface
# MAGIC    - Maps feedback_type to ground_truth automatically
# MAGIC    - Returns feedback_id
# MAGIC
# MAGIC 2. **`submit_feedback_batch()`** - Submit multiple feedbacks at once
# MAGIC    - Efficient for batch processing
# MAGIC    - Returns list of feedback_ids
# MAGIC
# MAGIC 3. **`get_prediction_info()`** - Look up what was predicted
# MAGIC    - Useful before submitting feedback
# MAGIC    - Shows model name, prediction, score
# MAGIC
# MAGIC 4. **`get_feedback_stats()`** - Check feedback collection metrics
# MAGIC    - Monitor feedback coverage
# MAGIC    - Identify gaps in feedback collection
# MAGIC
# MAGIC ### Feedback Types:
# MAGIC - `true-positive`: Model said PNEUMONIA, correct ✅
# MAGIC - `false-positive`: Model said PNEUMONIA, wrong ❌ (was NORMAL)
# MAGIC - `true-negative`: Model said NORMAL, correct ✅
# MAGIC - `false-negative`: Model said NORMAL, wrong ❌ (was PNEUMONIA)
# MAGIC
# MAGIC ### Next Steps:
# MAGIC 1. Deploy A/B testing endpoint (generates predictions with IDs)
# MAGIC 2. Use this collector to submit radiologist feedback
# MAGIC 3. Query `model_performance_live` view to see per-model accuracy
# MAGIC 4. Make data-driven decision: promote Challenger to Champion!
# MAGIC
# MAGIC **Educational Value**: Shows how feedback loops close the MLOps cycle!
