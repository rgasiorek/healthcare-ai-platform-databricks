-- Databricks Dashboard: Model Performance Comparison
-- This dashboard compares ML model performance using radiologist feedback

-- Configuration
-- Replace with your catalog name if different
SET catalog_name = 'healthcare_catalog_dev';

-- =============================================================================
-- Query 1: ML Metrics by Model
-- Calculates Precision, Recall, F1 Score, and Accuracy from radiologist feedback
-- =============================================================================

SELECT
  p.model_name,
  p.model_version,
  COUNT(DISTINCT f.feedback_id) as total_feedback,

  -- True Positives
  SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) as true_positives,

  -- False Positives
  SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END) as false_positives,

  -- True Negatives
  SUM(CASE WHEN f.feedback_type = 'true-negative' THEN 1 ELSE 0 END) as true_negatives,

  -- False Negatives
  SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END) as false_negatives,

  -- Precision: TP / (TP + FP)
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END),
      0
    ),
    4
  ) as precision,

  -- Recall (Sensitivity): TP / (TP + FN)
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END),
      0
    ),
    4
  ) as recall,

  -- Specificity: TN / (TN + FP)
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-negative' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-negative' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END),
      0
    ),
    4
  ) as specificity,

  -- F1 Score: 2 * (Precision * Recall) / (Precision + Recall)
  ROUND(
    2.0 *
    (
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
      NULLIF(
        SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
        SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END),
        0
      )
    ) *
    (
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
      NULLIF(
        SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
        SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END),
        0
      )
    ) /
    NULLIF(
      (
        SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
        NULLIF(
          SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
          SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END),
          0
        )
      ) +
      (
        SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
        NULLIF(
          SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
          SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END),
          0
        )
      ),
      0
    ),
    4
  ) as f1_score,

  -- Accuracy: (TP + TN) / (TP + TN + FP + FN)
  ROUND(
    (
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'true-negative' THEN 1 ELSE 0 END)
    ) * 1.0 /
    NULLIF(COUNT(DISTINCT f.feedback_id), 0),
    4
  ) as accuracy

FROM healthcare_catalog_dev.gold.pneumonia_predictions p
INNER JOIN healthcare_catalog_dev.gold.prediction_feedback f
  ON p.prediction_id = f.prediction_id
GROUP BY p.model_name, p.model_version
ORDER BY f1_score DESC;


-- =============================================================================
-- Query 2: Confusion Matrix
-- Shows the distribution of prediction outcomes
-- =============================================================================

SELECT
  f.feedback_type,
  COUNT(*) as count,
  ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2) as percentage
FROM healthcare_catalog_dev.gold.prediction_feedback f
GROUP BY f.feedback_type
ORDER BY
  CASE f.feedback_type
    WHEN 'true-positive' THEN 1
    WHEN 'false-positive' THEN 2
    WHEN 'false-negative' THEN 3
    WHEN 'true-negative' THEN 4
    ELSE 5
  END;


-- =============================================================================
-- Query 3: Confusion Matrix (2x2 Grid Format)
-- Formatted as a traditional confusion matrix
-- =============================================================================

SELECT
  'Predicted PNEUMONIA' as predicted_label,
  SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) as actual_pneumonia,
  SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END) as actual_normal
FROM healthcare_catalog_dev.gold.prediction_feedback f

UNION ALL

SELECT
  'Predicted NORMAL' as predicted_label,
  SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END) as actual_pneumonia,
  SUM(CASE WHEN f.feedback_type = 'true-negative' THEN 1 ELSE 0 END) as actual_normal
FROM healthcare_catalog_dev.gold.prediction_feedback f;


-- =============================================================================
-- Query 4: Prediction Confidence Analysis
-- Shows how confident the model was for correct vs incorrect predictions
-- =============================================================================

SELECT
  CASE
    WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 'Correct'
    ELSE 'Incorrect'
  END as prediction_outcome,
  ROUND(AVG(p.confidence_score), 4) as avg_confidence,
  ROUND(MIN(p.confidence_score), 4) as min_confidence,
  ROUND(MAX(p.confidence_score), 4) as max_confidence,
  COUNT(*) as count
FROM healthcare_catalog_dev.gold.pneumonia_predictions p
INNER JOIN healthcare_catalog_dev.gold.prediction_feedback f
  ON p.prediction_id = f.prediction_id
GROUP BY
  CASE
    WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 'Correct'
    ELSE 'Incorrect'
  END;


-- =============================================================================
-- Query 5: Performance Over Time
-- Shows how model performance changes over time
-- =============================================================================

SELECT
  DATE_TRUNC('day', f.timestamp) as date,
  COUNT(*) as predictions,

  -- Accuracy per day
  ROUND(
    (
      SUM(CASE WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 1 ELSE 0 END)
    ) * 100.0 / COUNT(*),
    2
  ) as accuracy_pct,

  -- Precision per day
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END),
      0
    ),
    4
  ) as precision,

  -- Recall per day
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END),
      0
    ),
    4
  ) as recall

FROM healthcare_catalog_dev.gold.prediction_feedback f
GROUP BY DATE_TRUNC('day', f.timestamp)
ORDER BY date DESC;


-- =============================================================================
-- Query 6: Radiologist Feedback Summary
-- Shows feedback submission patterns by radiologist
-- =============================================================================

SELECT
  f.radiologist_id,
  COUNT(*) as total_reviews,
  SUM(CASE WHEN f.confidence = 'confirmed' THEN 1 ELSE 0 END) as confirmed_reviews,
  SUM(CASE WHEN f.confidence = 'uncertain' THEN 1 ELSE 0 END) as uncertain_reviews,
  SUM(CASE WHEN f.confidence = 'needs_review' THEN 1 ELSE 0 END) as needs_review_count,

  -- Agreement with AI (true-positive + true-negative)
  ROUND(
    SUM(CASE WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2
  ) as ai_agreement_pct,

  MIN(f.timestamp) as first_review,
  MAX(f.timestamp) as last_review

FROM healthcare_catalog_dev.gold.prediction_feedback f
GROUP BY f.radiologist_id
ORDER BY total_reviews DESC;


-- =============================================================================
-- Query 7: Error Analysis - False Positives vs False Negatives
-- Detailed view of incorrect predictions
-- =============================================================================

SELECT
  f.feedback_type,
  COUNT(*) as count,
  ROUND(AVG(p.prediction_probability), 4) as avg_prediction_probability,
  ROUND(AVG(p.confidence_score), 4) as avg_confidence,

  -- Percentage of errors that are high confidence (>0.8)
  ROUND(
    SUM(CASE WHEN p.confidence_score > 0.8 THEN 1 ELSE 0 END) * 100.0 / COUNT(*),
    2
  ) as high_confidence_errors_pct

FROM healthcare_catalog_dev.gold.pneumonia_predictions p
INNER JOIN healthcare_catalog_dev.gold.prediction_feedback f
  ON p.prediction_id = f.prediction_id
WHERE f.feedback_type IN ('false-positive', 'false-negative')
GROUP BY f.feedback_type;


-- =============================================================================
-- Query 8: Prediction Coverage
-- Shows how many predictions have received feedback
-- =============================================================================

SELECT
  COUNT(DISTINCT p.prediction_id) as total_predictions,
  COUNT(DISTINCT f.feedback_id) as predictions_with_feedback,
  COUNT(DISTINCT p.prediction_id) - COUNT(DISTINCT f.feedback_id) as predictions_pending_feedback,
  ROUND(COUNT(DISTINCT f.feedback_id) * 100.0 / COUNT(DISTINCT p.prediction_id), 2) as feedback_coverage_pct
FROM healthcare_catalog_dev.gold.pneumonia_predictions p
LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
  ON p.prediction_id = f.prediction_id;
