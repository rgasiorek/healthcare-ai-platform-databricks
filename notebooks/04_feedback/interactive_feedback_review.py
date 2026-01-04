# Databricks notebook source
# MAGIC %md
# MAGIC # Interactive Feedback Review - Radiologist Workflow
# MAGIC
# MAGIC This notebook provides an interactive interface for radiologists to:
# MAGIC 1. View recent predictions made by the AI models
# MAGIC 2. See the actual X-ray images
# MAGIC 3. Provide ground truth diagnosis (NORMAL vs PNEUMONIA)
# MAGIC 4. Submit feedback to the feedback system
# MAGIC
# MAGIC **Use Case**: After doctors review X-rays, they confirm or correct the AI diagnosis

# COMMAND ----------
# MAGIC %md
# MAGIC ## Setup: Install Dependencies

# COMMAND ----------
%pip install Pillow --quiet

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Recent Predictions (Awaiting Feedback)

# COMMAND ----------
from pyspark.sql.functions import col, current_date
from datetime import datetime

# Query recent predictions that don't have feedback yet
recent_predictions = spark.sql("""
    SELECT
        p.request_id,
        p.served_model_name,
        p.timestamp,
        CAST(p.response:predictions[0][0] AS DOUBLE) as prediction_score,
        CASE
            WHEN CAST(p.response:predictions[0][0] AS DOUBLE) > 0.5
            THEN 'PNEUMONIA'
            ELSE 'NORMAL'
        END as predicted_diagnosis,
        ROUND(CAST(p.response:predictions[0][0] AS DOUBLE) * 100, 1) as confidence_pct,
        -- Try to find the original image path from the request payload
        -- This is a simplified version - in production you'd store image_id in the request
        ROW_NUMBER() OVER (ORDER BY p.timestamp DESC) as prediction_number
    FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
    LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
        ON p.request_id = f.prediction_id
    WHERE f.feedback_id IS NULL  -- Only predictions without feedback
      AND p.date >= current_date() - INTERVAL 7 DAYS
    ORDER BY p.timestamp DESC
    LIMIT 20
""")

# Collect to driver for interactive review
predictions_to_review = recent_predictions.collect()

print(f"Found {len(predictions_to_review)} predictions awaiting feedback")
print("=" * 80)

if len(predictions_to_review) == 0:
    print("⚠️  No predictions found. Run the demo notebook first to generate predictions.")
else:
    print(f"Most recent prediction:")
    first = predictions_to_review[0]
    print(f"  Request ID: {first.request_id}")
    print(f"  Model: {first.served_model_name}")
    print(f"  Predicted: {first.predicted_diagnosis}")
    print(f"  Confidence: {first.confidence_pct}%")
    print(f"  Timestamp: {first.timestamp}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Load Feedback Collector

# COMMAND ----------
import sys
sys.path.append('/Workspace/Shared')

from feedback_collector import submit_feedback

print("✅ Feedback collector loaded")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Interactive Feedback Interface
# MAGIC
# MAGIC **Instructions for Radiologists**:
# MAGIC 1. Review each prediction below
# MAGIC 2. Look at the model's diagnosis
# MAGIC 3. Provide your expert diagnosis (ground truth)
# MAGIC 4. Submit feedback
# MAGIC
# MAGIC **Note**: In a production system, you would:
# MAGIC - Display the actual X-ray image
# MAGIC - Have a web UI with proper forms
# MAGIC - Integrate with PACS/RIS systems

# COMMAND ----------
# MAGIC %md
# MAGIC ### Prediction #1 - Review and Feedback

# COMMAND ----------
if len(predictions_to_review) > 0:
    # Get first prediction
    pred = predictions_to_review[0]

    print("=" * 80)
    print("PREDICTION REVIEW #1")
    print("=" * 80)
    print(f"Request ID: {pred.request_id}")
    print(f"Timestamp: {pred.timestamp}")
    print(f"Model Used: {pred.served_model_name}")
    print()
    print(f"AI DIAGNOSIS: {pred.predicted_diagnosis}")
    print(f"Confidence: {pred.confidence_pct}%")
    print()
    print("📋 RADIOLOGIST: What is your diagnosis?")
    print()
    print("Options:")
    print("  1. NORMAL - No signs of pneumonia")
    print("  2. PNEUMONIA - Pneumonia confirmed")
else:
    print("No predictions to review. Run demo notebook first.")

# COMMAND ----------
# MAGIC %md
# MAGIC #### Submit Feedback for Prediction #1
# MAGIC
# MAGIC **Scenario Examples**:
# MAGIC - If AI said PNEUMONIA and you confirm PNEUMONIA → true-positive
# MAGIC - If AI said PNEUMONIA but it's actually NORMAL → false-positive
# MAGIC - If AI said NORMAL and you confirm NORMAL → true-negative
# MAGIC - If AI said NORMAL but it's actually PNEUMONIA → false-negative

# COMMAND ----------
# Example 1: AI was CORRECT - Confirmed PNEUMONIA
if len(predictions_to_review) > 0 and predictions_to_review[0].predicted_diagnosis == "PNEUMONIA":
    print("EXAMPLE: AI said PNEUMONIA, radiologist CONFIRMS it")
    print()

    feedback_id = submit_feedback(
        prediction_id=predictions_to_review[0].request_id,
        feedback_type="true-positive",  # AI correct: said PNEUMONIA, it IS pneumonia
        radiologist_id="DR001",
        confidence="confirmed",
        feedback_source="radiologist_review",
        notes="Clear consolidation in right lower lobe, consistent with bacterial pneumonia"
    )

    print(f"✅ Feedback submitted: {feedback_id}")
    print(f"   Prediction: {predictions_to_review[0].request_id}")
    print(f"   Ground Truth: PNEUMONIA")
    print(f"   Result: True Positive (AI was correct)")

elif len(predictions_to_review) > 0 and predictions_to_review[0].predicted_diagnosis == "NORMAL":
    print("EXAMPLE: AI said NORMAL, radiologist CONFIRMS it")
    print()

    feedback_id = submit_feedback(
        prediction_id=predictions_to_review[0].request_id,
        feedback_type="true-negative",  # AI correct: said NORMAL, it IS normal
        radiologist_id="DR001",
        confidence="confirmed",
        feedback_source="radiologist_review",
        notes="Clear lung fields, no abnormalities detected"
    )

    print(f"✅ Feedback submitted: {feedback_id}")
    print(f"   Prediction: {predictions_to_review[0].request_id}")
    print(f"   Ground Truth: NORMAL")
    print(f"   Result: True Negative (AI was correct)")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Prediction #2 - Example of AI Being WRONG

# COMMAND ----------
if len(predictions_to_review) > 1:
    # Get second prediction
    pred = predictions_to_review[1]

    print("=" * 80)
    print("PREDICTION REVIEW #2")
    print("=" * 80)
    print(f"Request ID: {pred.request_id}")
    print(f"Model Used: {pred.served_model_name}")
    print(f"AI DIAGNOSIS: {pred.predicted_diagnosis}")
    print(f"Confidence: {pred.confidence_pct}%")
    print()

    # Example: AI said PNEUMONIA but radiologist says NORMAL (false positive)
    if pred.predicted_diagnosis == "PNEUMONIA":
        print("SCENARIO: AI said PNEUMONIA, but radiologist says NORMAL")
        print()

        feedback_id = submit_feedback(
            prediction_id=pred.request_id,
            feedback_type="false-positive",  # AI wrong: said PNEUMONIA, actually NORMAL
            radiologist_id="DR001",
            confidence="confirmed",
            feedback_source="radiologist_review",
            notes="Artifact from patient movement, not infection. False positive."
        )

        print(f"✅ Feedback submitted: {feedback_id}")
        print(f"   Prediction: {pred.request_id}")
        print(f"   AI Said: PNEUMONIA")
        print(f"   Ground Truth: NORMAL")
        print(f"   Result: False Positive (AI was WRONG)")

    # Example: AI said NORMAL but radiologist says PNEUMONIA (false negative)
    elif pred.predicted_diagnosis == "NORMAL":
        print("SCENARIO: AI said NORMAL, but radiologist says PNEUMONIA")
        print()

        feedback_id = submit_feedback(
            prediction_id=pred.request_id,
            feedback_type="false-negative",  # AI wrong: said NORMAL, actually PNEUMONIA
            radiologist_id="DR002",
            confidence="confirmed",
            feedback_source="radiologist_review",
            notes="Subtle infiltrate missed by AI. Early-stage pneumonia."
        )

        print(f"✅ Feedback submitted: {feedback_id}")
        print(f"   Prediction: {pred.request_id}")
        print(f"   AI Said: NORMAL")
        print(f"   Ground Truth: PNEUMONIA")
        print(f"   Result: False Negative (AI was WRONG - CRITICAL MISS)")

# COMMAND ----------
# MAGIC %md
# MAGIC ### Batch Feedback Review (For Remaining Predictions)

# COMMAND ----------
print("=" * 80)
print("BATCH FEEDBACK REVIEW")
print("=" * 80)
print()
print(f"Predictions awaiting review: {len(predictions_to_review)}")
print()
print("In a production system, you would:")
print("  1. Display X-ray images side-by-side with AI prediction")
print("  2. Provide dropdown menus or buttons for feedback")
print("  3. Allow bulk review sessions")
print("  4. Track which radiologist reviewed which case")
print()
print("For this demo, we'll simulate reviewing all predictions:")

# COMMAND ----------
# Simulate batch feedback for demonstration
# In production, this would be an interactive UI
import random

if len(predictions_to_review) > 2:
    print("Simulating radiologist review of remaining predictions...\n")

    feedback_count = 0
    for i, pred in enumerate(predictions_to_review[2:8], start=3):  # Review up to 6 more
        # Simulate radiologist agreement/disagreement
        # 80% of the time, radiologist agrees with AI
        # 20% of the time, radiologist disagrees
        radiologist_agrees = random.random() < 0.8

        if pred.predicted_diagnosis == "PNEUMONIA":
            if radiologist_agrees:
                feedback_type = "true-positive"
                ground_truth = "PNEUMONIA"
                notes = "Confirmed pneumonia"
            else:
                feedback_type = "false-positive"
                ground_truth = "NORMAL"
                notes = "False alarm, actually normal"
        else:  # AI said NORMAL
            if radiologist_agrees:
                feedback_type = "true-negative"
                ground_truth = "NORMAL"
                notes = "Confirmed normal"
            else:
                feedback_type = "false-negative"
                ground_truth = "PNEUMONIA"
                notes = "AI missed early pneumonia"

        feedback_id = submit_feedback(
            prediction_id=pred.request_id,
            feedback_type=feedback_type,
            radiologist_id=f"DR00{random.randint(1, 5)}",
            confidence="confirmed",
            feedback_source="batch_review",
            notes=notes
        )

        result_symbol = "✅" if radiologist_agrees else "❌"
        print(f"{result_symbol} Prediction #{i}: AI={pred.predicted_diagnosis}, Truth={ground_truth}, Type={feedback_type}")
        feedback_count += 1

    print()
    print(f"✅ Submitted {feedback_count} feedback entries")
else:
    print("Need more predictions. Run demo notebook to generate more.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Verify Feedback Submission

# COMMAND ----------
# Query how many predictions now have feedback
feedback_summary = spark.sql("""
    SELECT
        'Total Predictions' as metric,
        COUNT(DISTINCT p.request_id) as count
    FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
    WHERE p.date >= current_date() - INTERVAL 7 DAYS

    UNION ALL

    SELECT
        'With Feedback' as metric,
        COUNT(DISTINCT f.prediction_id) as count
    FROM healthcare_catalog_dev.gold.prediction_feedback f
    WHERE f.timestamp >= current_date() - INTERVAL 7 DAYS

    UNION ALL

    SELECT
        'Coverage %' as metric,
        ROUND(
            COUNT(DISTINCT f.prediction_id) * 100.0 /
            (SELECT COUNT(DISTINCT request_id)
             FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions
             WHERE date >= current_date() - INTERVAL 7 DAYS),
            2
        ) as count
    FROM healthcare_catalog_dev.gold.prediction_feedback f
    WHERE f.timestamp >= current_date() - INTERVAL 7 DAYS
""")

print("FEEDBACK COVERAGE SUMMARY")
print("=" * 80)
feedback_summary.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## View Feedback Breakdown by Type

# COMMAND ----------
feedback_breakdown = spark.sql("""
    SELECT
        feedback_type,
        COUNT(*) as count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
    FROM healthcare_catalog_dev.gold.prediction_feedback
    WHERE timestamp >= current_date() - INTERVAL 7 DAYS
    GROUP BY feedback_type
    ORDER BY count DESC
""")

print("FEEDBACK TYPE BREAKDOWN")
print("=" * 80)
feedback_breakdown.show(truncate=False)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC ✅ **Feedback Submitted**: Ground truth labels now linked to predictions
# MAGIC
# MAGIC **What to do next**:
# MAGIC 1. Run `/Shared/monitor-ab-test` to see Champion vs Challenger performance
# MAGIC 2. Check SQL queries in EXECUTION_GUIDE.md for BI dashboard
# MAGIC 3. Query `model_performance_live` view for real-time accuracy
# MAGIC
# MAGIC **In Production**:
# MAGIC - Build web UI for radiologists (not just notebooks)
# MAGIC - Display actual X-ray images with DICOM viewer
# MAGIC - Integrate with hospital PACS/RIS systems
# MAGIC - Set up daily review workflows
# MAGIC - Track inter-rater reliability (multiple radiologists)

# COMMAND ----------
# MAGIC %md
# MAGIC ---
# MAGIC ## Appendix: Quick Feedback Helpers

# COMMAND ----------
# Helper function for quick feedback
def quick_feedback(request_id, ai_was_correct, actual_diagnosis):
    """
    Simplified feedback submission

    Args:
        request_id: The prediction request ID
        ai_was_correct: True if AI was correct, False if wrong
        actual_diagnosis: "NORMAL" or "PNEUMONIA"
    """
    # Get the AI prediction for this request
    pred_row = spark.sql(f"""
        SELECT
            CASE
                WHEN CAST(response:predictions[0][0] AS DOUBLE) > 0.5
                THEN 'PNEUMONIA'
                ELSE 'NORMAL'
            END as ai_diagnosis
        FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions
        WHERE request_id = '{request_id}'
        LIMIT 1
    """).collect()

    if not pred_row:
        print(f"❌ Request ID not found: {request_id}")
        return

    ai_diagnosis = pred_row[0].ai_diagnosis

    # Determine feedback type
    if ai_diagnosis == "PNEUMONIA" and actual_diagnosis == "PNEUMONIA":
        feedback_type = "true-positive"
    elif ai_diagnosis == "PNEUMONIA" and actual_diagnosis == "NORMAL":
        feedback_type = "false-positive"
    elif ai_diagnosis == "NORMAL" and actual_diagnosis == "NORMAL":
        feedback_type = "true-negative"
    elif ai_diagnosis == "NORMAL" and actual_diagnosis == "PNEUMONIA":
        feedback_type = "false-negative"
    else:
        print(f"❌ Invalid diagnosis: {actual_diagnosis}")
        return

    feedback_id = submit_feedback(
        prediction_id=request_id,
        feedback_type=feedback_type,
        radiologist_id="DR_QUICK",
        confidence="confirmed",
        feedback_source="quick_review"
    )

    print(f"✅ Feedback submitted: {feedback_id}")
    print(f"   AI: {ai_diagnosis}, Truth: {actual_diagnosis}")
    print(f"   Type: {feedback_type}")
    return feedback_id

# Example usage:
# quick_feedback("your-request-id-here", ai_was_correct=True, actual_diagnosis="PNEUMONIA")
