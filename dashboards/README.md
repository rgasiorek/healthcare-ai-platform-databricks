# Model Performance Dashboard

This dashboard compares ML model performance using radiologist feedback data from the gold layer.

## Data Sources

- `healthcare_catalog_dev.gold.pneumonia_predictions` - AI predictions from models
- `healthcare_catalog_dev.gold.prediction_feedback` - Radiologist ground truth feedback

## Metrics Displayed

### 1. ML Metrics by Model
- **Precision**: TP / (TP + FP) - How many predicted positives are actually positive
- **Recall (Sensitivity)**: TP / (TP + FN) - How many actual positives were detected
- **Specificity**: TN / (TN + FP) - How many actual negatives were correctly identified
- **F1 Score**: Harmonic mean of precision and recall
- **Accuracy**: (TP + TN) / Total - Overall correctness

### 2. Confusion Matrix
Shows distribution of:
- True Positives (Correctly identified pneumonia)
- False Positives (Incorrectly identified pneumonia)
- True Negatives (Correctly identified normal)
- False Negatives (Missed pneumonia cases)

### 3. Performance Over Time
Tracks accuracy, precision, and recall trends by day

### 4. Confidence Analysis
Compares model confidence for correct vs incorrect predictions

### 5. Error Analysis
Detailed view of false positives vs false negatives with confidence metrics

### 6. Radiologist Feedback Summary
Shows review patterns and AI agreement rates by radiologist

### 7. Prediction Coverage
Tracks what percentage of predictions have received feedback

## How to Create the Dashboard in Databricks

### Method 1: Databricks UI (Recommended)

1. **Navigate to SQL in Databricks workspace**
   - Click "SQL" in the left sidebar
   - Click "Dashboards"
   - Click "Create Dashboard"

2. **Create Dashboard**
   - Name: "Model Performance Comparison"
   - Description: "Compare Keras vs PyTorch pneumonia classifier performance"

3. **Add Visualizations** (one per query):

   **Visualization 1: ML Metrics Table**
   - Click "Add" → "Visualization"
   - Copy Query 1 from `model_comparison_dashboard.sql`
   - Visualization type: **Table**
   - Title: "ML Metrics by Model"

   **Visualization 2: Confusion Matrix (Bar Chart)**
   - Click "Add" → "Visualization"
   - Copy Query 2 from `model_comparison_dashboard.sql`
   - Visualization type: **Bar Chart**
   - X-axis: `feedback_type`
   - Y-axis: `count`
   - Title: "Confusion Matrix Distribution"

   **Visualization 3: Confusion Matrix (Grid)**
   - Click "Add" → "Visualization"
   - Copy Query 3 from `model_comparison_dashboard.sql`
   - Visualization type: **Table**
   - Title: "Confusion Matrix (2x2)"

   **Visualization 4: Confidence Analysis**
   - Click "Add" → "Visualization"
   - Copy Query 4 from `model_comparison_dashboard.sql`
   - Visualization type: **Bar Chart**
   - X-axis: `prediction_outcome`
   - Y-axis: `avg_confidence`
   - Title: "Confidence by Outcome"

   **Visualization 5: Performance Over Time**
   - Click "Add" → "Visualization"
   - Copy Query 5 from `model_comparison_dashboard.sql`
   - Visualization type: **Line Chart**
   - X-axis: `date`
   - Y-axes: `accuracy_pct`, `precision`, `recall`
   - Title: "Performance Trends"

   **Visualization 6: Radiologist Summary**
   - Click "Add" → "Visualization"
   - Copy Query 6 from `model_comparison_dashboard.sql`
   - Visualization type: **Table**
   - Title: "Radiologist Feedback Summary"

   **Visualization 7: Error Analysis**
   - Click "Add" → "Visualization"
   - Copy Query 7 from `model_comparison_dashboard.sql`
   - Visualization type: **Bar Chart**
   - X-axis: `feedback_type`
   - Y-axis: `count`
   - Title: "False Positives vs False Negatives"

   **Visualization 8: Coverage Counter**
   - Click "Add" → "Visualization"
   - Copy Query 8 from `model_comparison_dashboard.sql`
   - Visualization type: **Counter**
   - Value: `feedback_coverage_pct`
   - Title: "Feedback Coverage %"

4. **Arrange Layout**
   - Drag and resize visualizations
   - Suggested layout:
     ```
     ┌─────────────────────────────────────────────┐
     │  ML Metrics by Model (Table)                │
     ├──────────────────┬──────────────────────────┤
     │  Confusion       │  Confusion Matrix        │
     │  Matrix (Bar)    │  (2x2 Grid)              │
     ├──────────────────┴──────────────────────────┤
     │  Performance Over Time (Line Chart)         │
     ├──────────────────┬──────────────────────────┤
     │  Confidence      │  Error Analysis          │
     │  Analysis        │  (FP vs FN)              │
     ├──────────────────┴──────────────────────────┤
     │  Radiologist Feedback Summary (Table)       │
     ├──────────────────────────────────────────────┤
     │  Coverage: XX%   │  (Counter)               │
     └──────────────────────────────────────────────┘
     ```

5. **Configure Refresh**
   - Click "Schedule" → Set refresh frequency (e.g., every hour)
   - For real-time monitoring, set to 5-15 minutes

6. **Share Dashboard**
   - Click "Share" button
   - Add users or groups
   - Set permissions (View/Edit)

### Method 2: Using SQL Queries Directly

If you prefer to run queries ad-hoc:

1. Open "SQL Editor" in Databricks
2. Copy any query from `model_comparison_dashboard.sql`
3. Run the query
4. View results in table or chart format
5. Save as a "Query" for reuse

### Method 3: Programmatic Creation (Advanced)

Use the Databricks REST API or SDK to create dashboards programmatically:

```python
# See create_dashboard.py (if created)
```

## Query Descriptions

- **Query 1**: Core ML metrics (Precision, Recall, F1, Accuracy) by model
- **Query 2**: Confusion matrix as simple counts
- **Query 3**: Confusion matrix in traditional 2x2 grid format
- **Query 4**: Confidence analysis for correct vs incorrect predictions
- **Query 5**: Performance trends over time (daily aggregation)
- **Query 6**: Radiologist feedback patterns and AI agreement rates
- **Query 7**: Detailed error analysis (FP vs FN with confidence)
- **Query 8**: Prediction coverage (how many predictions have feedback)

## Notes

- Dashboard requires feedback data to display metrics
- Run `generate_sample_predictions.py` to create predictions
- Run `end_to_end_demo.py` Step 4 to submit feedback
- Metrics are calculated ONLY from radiologist feedback (not dataset labels)

## Interpreting Results

### Good Model Performance
- **Precision > 0.85**: Few false alarms
- **Recall > 0.90**: Catches most pneumonia cases (critical for healthcare)
- **F1 Score > 0.87**: Good balance
- **Low False Negative Rate**: Missing pneumonia is dangerous

### Red Flags
- **High False Negatives**: Model missing pneumonia cases (dangerous)
- **High False Positives**: Unnecessary follow-up procedures (costly)
- **Low Confidence on Correct Predictions**: Model is guessing
- **High Confidence on Incorrect Predictions**: Overconfident mistakes

## Updating the Dashboard

To add new visualizations:
1. Write a new SQL query using the gold layer tables
2. Add it to `model_comparison_dashboard.sql`
3. Create a visualization in Databricks dashboard UI
4. Update this README with the new query description
