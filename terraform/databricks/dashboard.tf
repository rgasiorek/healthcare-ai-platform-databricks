# Databricks SQL Dashboard - Model Performance Monitoring
# Displays real-time ML metrics from radiologist feedback

# Data source for current user
data "databricks_current_user" "me" {}

# Dashboard container
resource "databricks_sql_dashboard" "model_performance" {
  name = "Model Performance Comparison"

  tags = [
    "mlops",
    "monitoring",
    "healthcare"
  ]
}

# Query 1: ML Metrics by Model
resource "databricks_sql_query" "ml_metrics" {
  data_source_id = databricks_sql_endpoint.healthcare_warehouse.data_source_id
  name           = "ML Metrics by Model"

  query = <<-EOT
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
  EOT
}

# Visualization 1: ML Metrics Table
resource "databricks_sql_visualization" "ml_metrics_table" {
  query_id    = databricks_sql_query.ml_metrics.id
  type        = "TABLE"
  name        = "ML Metrics Table"
  description = "Precision, Recall, F1 Score, Accuracy by model"

  options = jsonencode({
    "version" : 2,
    "itemsPerPage" : 25,
    "paginationSize" : "default"
  })
}

# Widget 1: Add visualization to dashboard
resource "databricks_sql_widget" "ml_metrics_widget" {
  dashboard_id     = databricks_sql_dashboard.model_performance.id
  visualization_id = databricks_sql_visualization.ml_metrics_table.id

  position {
    size_x = 6
    size_y = 8
    pos_x  = 0
    pos_y  = 0
  }
}

# Query 2: Confusion Matrix
resource "databricks_sql_query" "confusion_matrix" {
  data_source_id = databricks_sql_endpoint.healthcare_warehouse.data_source_id
  name           = "Confusion Matrix Distribution"

  query = <<-EOT
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
  EOT
}

# Visualization 2: Confusion Matrix Bar Chart
resource "databricks_sql_visualization" "confusion_matrix_bar" {
  query_id    = databricks_sql_query.confusion_matrix.id
  type        = "CHART"
  name        = "Confusion Matrix"
  description = "Distribution of prediction outcomes"

  options = jsonencode({
    "version" : 2,
    "globalSeriesType" : "column",
    "sortX" : true,
    "legend" : { "enabled" : true },
    "xAxis" : { "type" : "category", "labels" : { "enabled" : true } },
    "yAxis" : [{ "type" : "linear" }],
    "series" : {
      "stacking" : null
    },
    "seriesOptions" : {
      "count" : { "yAxis" : 0, "type" : "column" }
    },
    "columnMapping" : {
      "feedback_type" : "x",
      "count" : "y"
    }
  })
}

# Widget 2: Confusion Matrix
resource "databricks_sql_widget" "confusion_matrix_widget" {
  dashboard_id     = databricks_sql_dashboard.model_performance.id
  visualization_id = databricks_sql_visualization.confusion_matrix_bar.id

  position {
    size_x = 3
    size_y = 8
    pos_x  = 6
    pos_y  = 0
  }
}

# Query 3: Performance Over Time
resource "databricks_sql_query" "performance_over_time" {
  data_source_id = databricks_sql_endpoint.healthcare_warehouse.data_source_id
  name           = "Performance Over Time"

  query = <<-EOT
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
  EOT
}

# Visualization 3: Performance Trends Line Chart
resource "databricks_sql_visualization" "performance_trends" {
  query_id    = databricks_sql_query.performance_over_time.id
  type        = "CHART"
  name        = "Performance Trends"
  description = "Accuracy, Precision, Recall over time"

  options = jsonencode({
    "version" : 2,
    "globalSeriesType" : "line",
    "sortX" : true,
    "legend" : { "enabled" : true },
    "xAxis" : { "type" : "datetime", "labels" : { "enabled" : true } },
    "yAxis" : [{ "type" : "linear", "title" : { "text" : "Percentage" } }],
    "series" : {
      "stacking" : null
    },
    "seriesOptions" : {
      "accuracy_pct" : { "yAxis" : 0, "type" : "line", "name" : "Accuracy %" },
      "precision_pct" : { "yAxis" : 0, "type" : "line", "name" : "Precision %" },
      "recall_pct" : { "yAxis" : 0, "type" : "line", "name" : "Recall %" }
    },
    "columnMapping" : {
      "date" : "x",
      "accuracy_pct" : "y",
      "precision_pct" : "y",
      "recall_pct" : "y"
    }
  })
}

# Widget 3: Performance Trends
resource "databricks_sql_widget" "performance_trends_widget" {
  dashboard_id     = databricks_sql_dashboard.model_performance.id
  visualization_id = databricks_sql_visualization.performance_trends.id

  position {
    size_x = 9
    size_y = 8
    pos_x  = 0
    pos_y  = 8
  }
}

# Query 4: Prediction Coverage Counter
resource "databricks_sql_query" "prediction_coverage" {
  data_source_id = databricks_sql_endpoint.healthcare_warehouse.data_source_id
  name           = "Prediction Coverage"

  query = <<-EOT
    SELECT
      COUNT(DISTINCT p.prediction_id) as total_predictions,
      COUNT(DISTINCT f.feedback_id) as predictions_with_feedback,
      ROUND(COUNT(DISTINCT f.feedback_id) * 100.0 / COUNT(DISTINCT p.prediction_id), 2) as feedback_coverage_pct
    FROM healthcare_catalog_dev.gold.pneumonia_predictions p
    LEFT JOIN healthcare_catalog_dev.gold.prediction_feedback f
      ON p.prediction_id = f.prediction_id
  EOT
}

# Visualization 4: Coverage Counter
resource "databricks_sql_visualization" "coverage_counter" {
  query_id    = databricks_sql_query.prediction_coverage.id
  type        = "COUNTER"
  name        = "Feedback Coverage"
  description = "Percentage of predictions with radiologist feedback"

  options = jsonencode({
    "counterLabel" : "Feedback Coverage",
    "counterColName" : "feedback_coverage_pct",
    "rowNumber" : 1,
    "targetRowNumber" : 1,
    "stringDecimal" : 1,
    "stringDecChar" : ".",
    "stringThouSep" : ",",
    "tooltipFormat" : "0,0.00"
  })
}

# Widget 4: Coverage Counter
resource "databricks_sql_widget" "coverage_counter_widget" {
  dashboard_id     = databricks_sql_dashboard.model_performance.id
  visualization_id = databricks_sql_visualization.coverage_counter.id

  position {
    size_x = 3
    size_y = 4
    pos_x  = 9
    pos_y  = 8
  }
}

# Outputs
output "dashboard_url" {
  value       = "https://${data.databricks_current_user.me.workspace_url}/sql/dashboards/${databricks_sql_dashboard.model_performance.id}"
  description = "URL to access the Model Performance Dashboard"
}
