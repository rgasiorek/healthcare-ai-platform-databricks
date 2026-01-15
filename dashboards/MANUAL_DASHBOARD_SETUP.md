# Manual Dashboard Setup (Databricks SQL)

Since Databricks has deprecated legacy SQL dashboard creation via API/Terraform, you need to create the dashboard manually in the UI.

## Why Manual Setup?

- **Legacy SQL Dashboards**: API creation disabled by Databricks
- **Lakeview Dashboards**: Require different resource type (`databricks_dashboard`), different structure
- **Simplest Solution**: Create manually in UI using SQL queries below

## Step-by-Step Instructions

### 1. Navigate to SQL Dashboards

1. Open your Databricks workspace
2. Click **SQL** in the left sidebar
3. Click **Dashboards**
4. Click **Create Dashboard**
5. Name: "Model Performance Comparison"
6. Click **Create**

### 2. Add Visualizations

For each query below, follow these steps:
1. Click **Add** → **Visualization**
2. Paste the SQL query
3. Click **Run**
4. Configure visualization settings (type, axes, etc.)
5. Click **Save**
6. Resize and position the widget

---

## Query 1: ML Metrics Table

**Visualization Type**: Table

```sql
SELECT
  p.model_name,
  p.model_version,
  COUNT(DISTINCT f.feedback_id) as total_feedback,
  SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) as true_positives,
  SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END) as false_positives,
  SUM(CASE WHEN f.feedback_type = 'true-negative' THEN 1 ELSE 0 END) as true_negatives,
  SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END) as false_negatives,
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END),
      0
    ),
    4
  ) as precision,
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END),
      0
    ),
    4
  ) as recall,
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
ORDER BY f1_score DESC
```

**Suggested Layout**: Top left, 6 columns wide, 8 rows tall

---

## Query 2: Confusion Matrix

**Visualization Type**: Bar Chart

```sql
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
  END
```

**Chart Settings**:
- X-axis: `feedback_type`
- Y-axis: `count`
- Chart type: Column (vertical bar)

**Suggested Layout**: Top right, 3 columns wide, 8 rows tall

---

## Query 3: Performance Over Time

**Visualization Type**: Line Chart

```sql
SELECT
  DATE_TRUNC('day', f.timestamp) as date,
  COUNT(*) as predictions,
  ROUND(
    (
      SUM(CASE WHEN f.feedback_type IN ('true-positive', 'true-negative') THEN 1 ELSE 0 END)
    ) * 100.0 / COUNT(*),
    2
  ) as accuracy_pct,
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-positive' THEN 1 ELSE 0 END),
      0
    ) * 100,
    2
  ) as precision_pct,
  ROUND(
    SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) * 1.0 /
    NULLIF(
      SUM(CASE WHEN f.feedback_type = 'true-positive' THEN 1 ELSE 0 END) +
      SUM(CASE WHEN f.feedback_type = 'false-negative' THEN 1 ELSE 0 END),
      0
    ) * 100,
    2
  ) as recall_pct
FROM healthcare_catalog_dev.gold.prediction_feedback f
GROUP BY DATE_TRUNC('day', f.timestamp)
ORDER BY date DESC
```

**Chart Settings**:
- X-axis: `date`
- Y-axis: `accuracy_pct`, `precision_pct`, `recall_pct`
- Chart type: Line
- Enable legend

**Suggested Layout**: Middle row, 9 columns wide, 8 rows tall

---

## Query 4: Prediction Coverage

**Visualization Type**: Counter

```sql
SELECT
  COUNT(DISTINCT p.prediction_id) as total_predictions,
  COUNT(DISTINCT f.feedback_id) as predictions_with_feedback,
  ROUND(COUNT(DISTINCT f.feedback_id) * 100.0 / COUNT(DISTINCT p.prediction_id), 2) as feedback_coverage_pct
FROM healthcare_catalog_dev.gold.pneumonia_predictions p
LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
  ON p.prediction_id = f.prediction_id
```

**Counter Settings**:
- Display field: `feedback_coverage_pct`
- Label: "Feedback Coverage %"

**Suggested Layout**: Bottom right, 3 columns wide, 4 rows tall

---

## Final Dashboard Layout

```
┌─────────────────────────────┬──────────────┐
│  ML Metrics Table           │  Confusion   │
│  (6 cols x 8 rows)          │  Matrix      │
│                             │  (3x8)       │
├─────────────────────────────┴──────────────┤
│  Performance Trends (Line Chart)           │
│  (9 cols x 8 rows)                         │
├────────────────────────────────┬───────────┤
│                                │ Coverage  │
│                                │ Counter   │
│                                │ (3x4)     │
└────────────────────────────────┴───────────┘
```

## Auto-Refresh

1. Click dashboard settings (⚙️)
2. Set refresh interval: 5 minutes (for near real-time monitoring)
3. Click **Save**

## Sharing

1. Click **Share** button
2. Add users or groups
3. Set permissions (View/Edit)

---

## Notes

- Dashboard requires data in both gold tables to display results
- Run `generate_sample_predictions.py` notebook to create test data
- Run `end_to_end_demo.py` notebook Step 4 to submit feedback
- Or use the Streamlit app for interactive feedback submission

## Future: Lakeview Dashboards

To migrate to modern Lakeview dashboards in the future:
- Use `databricks_dashboard` Terraform resource (not `databricks_sql_dashboard`)
- Different structure, requires dashboard definition JSON
- More features: filters, parameters, better layouts
