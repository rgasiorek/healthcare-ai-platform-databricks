# Execution Guide - End-to-End Run

> **Goal**: Run everything sequentially to see complete MLOps results in BI dashboard

---

## Prerequisites (Already Done)

✅ **Infrastructure deployed** via `terraform apply`
✅ **Notebooks uploaded** to Databricks `/Shared/`
✅ **AWS/Databricks/GitHub access** configured

---

## Execution Sequence

### Step 1: Data Ingestion (10-15 minutes)
**What**: Download 1000 X-ray images from Kaggle → Delta tables

**Run**:
1. Go to Databricks Workspace → **Shared** → `ingest-kaggle-xray-data`
2. Attach to cluster: `healthcare-data-cluster-dev`
3. Click **Run All**
4. Wait ~10-15 minutes

**Verify**:
```sql
SELECT category, COUNT(*) as count
FROM healthcare_catalog_dev.bronze.kaggle_xray_metadata
GROUP BY category;

-- Expected: 500 NORMAL, 500 PNEUMONIA
```

**What you get**:
- ✅ 1000 JPEG images in `/Volumes/healthcare_catalog_dev/bronze/xray_images/`
- ✅ 1000 rows in `bronze.kaggle_xray_metadata` table

---

### Step 2: Train Champion Model (5-10 minutes)
**What**: Train TensorFlow/Keras CNN on 100 images

**Run**:
1. Go to **Shared** → `train-poc-model`
2. Attach to cluster: `healthcare-data-cluster-dev`
3. Click **Run All**
4. Wait ~5-10 minutes

**Verify**:
- Check MLflow UI: **Machine Learning** → **Experiments** → `/Shared/pneumonia-poc-experiments`
- Check Model Registry: **Machine Learning** → **Models** → `pneumonia_poc_classifier`

**What you get**:
- ✅ Trained Keras model registered as `pneumonia_poc_classifier` version 1
- ✅ Experiment tracking with metrics (accuracy, loss)

---

### Step 3: Train Challenger Model (5-10 minutes)
**What**: Train PyTorch CNN on 100 images (same architecture)

**Run**:
1. Go to **Shared** → `train-poc-model-pytorch`
2. Attach to cluster: `healthcare-data-cluster-dev`
3. Click **Run All**
4. Wait ~5-10 minutes

**Verify**:
- Check MLflow UI: **Machine Learning** → **Experiments** → `/Shared/pneumonia-pytorch-poc-experiments`
- Check Model Registry: **Machine Learning** → **Models** → `pneumonia_poc_classifier_pytorch`

**What you get**:
- ✅ Trained PyTorch model registered as `pneumonia_poc_classifier_pytorch` version 1
- ✅ Two models ready for A/B testing

---

### Step 4: Deploy A/B Testing Endpoint (5-10 minutes)
**What**: Create single endpoint serving both models with 50/50 traffic split

**Run**:
1. Go to **Shared** → `deploy-ab-testing-endpoint`
2. Attach to cluster: `healthcare-data-cluster-dev`
3. Click **Run All**
4. Wait ~5-10 minutes for endpoint to be READY

**Verify**:
- Go to **Serving** → Check `pneumonia-classifier-ab-test`
- Status should be **READY** (green)

**What you get**:
- ✅ REST API endpoint: `https://<workspace>/serving-endpoints/pneumonia-classifier-ab-test/invocations`
- ✅ Traffic split: 50% Keras, 50% PyTorch
- ✅ Inference logging enabled (auto_capture)

---

### Step 5: Make Predictions (2-3 minutes)
**What**: Test the A/B endpoint, generate predictions from both models

**Run**:
1. Go to **Shared** → `demo-model-usage`
2. Attach to cluster: `healthcare-data-cluster-dev`
3. **Run cells up to and including "APPROACH 2: REST API"**
4. Note: This will make ~5-10 predictions via REST API

**What you get**:
- ✅ 5-10 predictions made
- ✅ Each prediction logged to `gold.pneumonia_classifier_predictions` table
- ✅ Request IDs captured for feedback

**Important**:
- Some predictions served by Keras model
- Some predictions served by PyTorch model
- `served_model_name` column tracks which model

---

### Step 6: Submit Feedback - Interactive Radiologist Review (2-3 minutes)
**What**: Review predictions and submit ground truth feedback (like a real radiologist)

**Run**:
1. Go to **Shared** → `interactive-feedback-review`
2. Attach to cluster: `healthcare-data-cluster-dev`
3. Click **Run All**

**What this notebook does**:
- Loads recent predictions that don't have feedback yet
- Shows AI prediction vs confidence
- Allows you to submit ground truth diagnosis
- Provides examples of:
  - **True Positive**: AI said PNEUMONIA, radiologist confirms PNEUMONIA
  - **False Positive**: AI said PNEUMONIA, radiologist says NORMAL
  - **True Negative**: AI said NORMAL, radiologist confirms NORMAL
  - **False Negative**: AI said NORMAL, radiologist says PNEUMONIA
- Submits feedback to `prediction_feedback` table
- Shows coverage statistics

**Interactive Workflow**:
```
For each prediction:
├─ View: Request ID, Model used, AI prediction, Confidence
├─ Review: What AI said (NORMAL or PNEUMONIA)
├─ Decide: What is the ground truth? (radiologist expertise)
└─ Submit: Feedback automatically categorized (TP/FP/TN/FN)
```

**Helper Function Included**:
```python
# Quick feedback submission
quick_feedback(
    request_id="abc-123",
    ai_was_correct=True,
    actual_diagnosis="PNEUMONIA"
)
```

**What you get**:
- ✅ Ground truth labels in `gold.prediction_feedback` table
- ✅ Link between predictions and actual diagnoses
- ✅ Automatic categorization (true-positive, false-positive, etc.)
- ✅ Feedback coverage statistics
- ✅ Ready for monitoring dashboard analysis

---

### Step 7: Monitor Performance (1-2 minutes)
**What**: Run monitoring dashboard to compare Champion vs Challenger

**Run**:
1. Go to **Shared** → `monitor-ab-test`
2. Attach to cluster: `healthcare-data-cluster-dev`
3. Click **Run All**

**What you get**:
- ✅ **Traffic Distribution Chart**: See actual 50/50 split
- ✅ **Feedback Coverage**: % of predictions with ground truth
- ✅ **Accuracy Comparison**: Keras vs PyTorch performance
- ✅ **Confusion Matrices**: TP/FP/TN/FN per model
- ✅ **Daily Trends**: Performance over time
- ✅ **Statistical Test**: Chi-square p-value
- ✅ **Recommendation**: PROMOTE / KEEP TESTING / ROLLBACK

---

### Step 8: Query Results in SQL Editor (BI Dashboard)
**What**: See all results in Databricks SQL for BI dashboards

**Run SQL Queries**:

#### 8.1 - Traffic Distribution
```sql
SELECT
    served_model_name,
    COUNT(*) as total_predictions,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as traffic_pct
FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions
WHERE date >= current_date() - INTERVAL 7 DAYS
GROUP BY served_model_name
ORDER BY total_predictions DESC;
```

#### 8.2 - Model Accuracy Comparison
```sql
SELECT
    p.served_model_name,
    COUNT(*) as total_with_feedback,
    SUM(CASE WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 1 ELSE 0 END) as correct,
    ROUND(AVG(CASE WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 1.0 ELSE 0.0 END) * 100, 2) as accuracy_pct
FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
INNER JOIN healthcare_catalog_dev.gold.prediction_feedback f
    ON p.request_id = f.prediction_id
WHERE p.date >= current_date() - INTERVAL 7 DAYS
GROUP BY p.served_model_name
ORDER BY accuracy_pct DESC;
```

#### 8.3 - Confusion Matrix (Both Models)
```sql
SELECT
    served_model_name,
    SUM(CASE WHEN feedback_type = 'true-positive' THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN feedback_type = 'false-positive' THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN feedback_type = 'true-negative' THEN 1 ELSE 0 END) as true_negatives,
    SUM(CASE WHEN feedback_type = 'false-negative' THEN 1 ELSE 0 END) as false_negatives
FROM healthcare_catalog_dev.gold.model_performance_live
WHERE prediction_date >= current_date() - INTERVAL 7 DAYS
GROUP BY served_model_name;
```

#### 8.4 - Daily Performance Trends
```sql
SELECT
    DATE(p.timestamp) as prediction_date,
    p.served_model_name,
    COUNT(*) as predictions_count,
    COUNT(f.feedback_id) as feedback_count,
    ROUND(AVG(CASE WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 1.0 ELSE 0.0 END) * 100, 2) as daily_accuracy
FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
    ON p.request_id = f.prediction_id
WHERE p.date >= current_date() - INTERVAL 7 DAYS
GROUP BY DATE(p.timestamp), p.served_model_name
ORDER BY prediction_date DESC, served_model_name;
```

#### 8.5 - Feedback Coverage
```sql
SELECT
    COUNT(DISTINCT p.request_id) as total_predictions,
    COUNT(DISTINCT f.feedback_id) as predictions_with_feedback,
    ROUND(COUNT(DISTINCT f.feedback_id) * 100.0 / COUNT(DISTINCT p.request_id), 2) as coverage_pct
FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions p
LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
    ON p.request_id = f.prediction_id
WHERE p.date >= current_date() - INTERVAL 7 DAYS;
```

---

## What You Should See (Expected Results)

### In SQL Editor / BI Dashboard:

**1. Traffic Distribution**:
```
served_model_name                                    | total_predictions | traffic_pct
----------------------------------------------------|-------------------|-------------
pneumonia_poc_classifier-1                          | 5                 | 50.0
pneumonia_poc_classifier_pytorch-1                  | 5                 | 50.0
```

**2. Model Accuracy** (after submitting feedback):
```
served_model_name                                    | total_with_feedback | correct | accuracy_pct
----------------------------------------------------|---------------------|---------|-------------
pneumonia_poc_classifier_pytorch-1                  | 3                   | 2       | 66.67
pneumonia_poc_classifier-1                          | 3                   | 2       | 66.67
```

**3. Confusion Matrix**:
```
served_model_name                | TP | FP | TN | FN
---------------------------------|----|----|----|----
pneumonia_poc_classifier-1       | 1  | 1  | 1  | 0
pneumonia_poc_classifier_pytorch | 1  | 0  | 2  | 0
```

**4. Recommendation** (from monitor notebook):
```
RECOMMENDATION: KEEP TESTING
- Sample size too small (< 30 per model)
- No statistically significant difference (p > 0.05)
- Continue collecting feedback
```

---

## Complete MLOps Cycle Demonstrated

At this point, you have:

✅ **1. TRAIN**: Two models trained (Keras + PyTorch)
✅ **2. DEPLOY**: A/B testing endpoint with traffic splitting
✅ **3. PREDICT**: Predictions made via REST API
✅ **4. COLLECT**: Feedback submitted with ground truth
✅ **5. ANALYZE**: Performance compared in SQL/BI dashboard
✅ **6. DECIDE**: Statistical test + recommendation (promote/keep/rollback)

---

## Creating a BI Dashboard (Optional)

### Databricks SQL Dashboard

1. Go to **SQL** → **Dashboards** → **Create Dashboard**
2. Add visualizations:
   - **Bar Chart**: Traffic distribution (Query 8.1)
   - **Gauge**: Feedback coverage (Query 8.5)
   - **Table**: Accuracy comparison (Query 8.2)
   - **Heatmap**: Confusion matrices (Query 8.3)
   - **Line Chart**: Daily accuracy trends (Query 8.4)

3. Set refresh schedule (hourly/daily)
4. Share with team

### Example Dashboard Layout:
```
┌─────────────────────────────────────────────────────────┐
│         Healthcare AI - A/B Testing Dashboard           │
├───────────────────┬─────────────────────────────────────┤
│ Traffic Split     │ Feedback Coverage                   │
│ [Bar Chart]       │ [Gauge: 60%]                        │
├───────────────────┴─────────────────────────────────────┤
│ Model Accuracy Comparison                               │
│ ┌─────────────────┬───────┬──────────┐                  │
│ │ Model           │ Total │ Accuracy │                  │
│ ├─────────────────┼───────┼──────────┤                  │
│ │ PyTorch         │ 3     │ 66.67%   │                  │
│ │ Keras           │ 3     │ 66.67%   │                  │
│ └─────────────────┴───────┴──────────┘                  │
├──────────────────────────────────────────────────────────┤
│ Confusion Matrices                                       │
│ [Heatmap: TP/FP/TN/FN for both models]                  │
├──────────────────────────────────────────────────────────┤
│ Daily Accuracy Trends                                    │
│ [Line Chart: Both models over time]                     │
└──────────────────────────────────────────────────────────┘
```

---

## Troubleshooting

### Issue: No predictions in inference table
**Cause**: Inference logging takes 1-2 minutes to appear
**Solution**: Wait 2-3 minutes, then run:
```sql
SELECT COUNT(*) FROM healthcare_catalog_dev.gold.pneumonia_classifier_predictions;
```

### Issue: Endpoint not READY
**Cause**: Model deployment takes 5-10 minutes
**Solution**: Check **Serving** UI, wait for green "READY" status

### Issue: No feedback data
**Cause**: Haven't submitted feedback yet
**Solution**: Run Step 6 (submit feedback) or insert manually via SQL

### Issue: All predictions from one model
**Cause**: Traffic split may take a few requests to balance
**Solution**: Make 10+ predictions to see distribution

---

## Next Steps

### Experiment Further:
1. **Adjust Traffic Split**: Change to 70/30, 90/10
2. **Submit More Feedback**: Get statistically significant results (30+ per model)
3. **Promote Winner**: Update endpoint to give 100% traffic to better model
4. **Add More Models**: Train a third model (different architecture)
5. **Automate Retraining**: Use feedback to retrain models

### Production Improvements:
1. **Larger Dataset**: Train on full 5000+ images
2. **Better Architecture**: Use transfer learning (EfficientNet, ResNet)
3. **Hyperparameter Tuning**: Use Databricks AutoML
4. **CI/CD Pipeline**: Automate deployment via GitHub Actions
5. **Alerting**: Set up notifications for model drift

---

## Summary

**Total Time**: ~30-40 minutes
**Result**: Complete MLOps platform with A/B testing visible in BI dashboard

**You've demonstrated**:
- Data ingestion (Kaggle → Delta)
- Model training (TensorFlow + PyTorch)
- Model serving (A/B testing endpoint)
- Inference logging (auto-capture)
- Feedback loop (ground truth collection)
- Performance monitoring (SQL/BI dashboard)
- Decision support (promote/keep/rollback)

**All without clicking in AWS/Databricks consoles - 100% automated workflow.**
