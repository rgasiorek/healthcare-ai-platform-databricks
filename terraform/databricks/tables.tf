# Production-ready Delta Tables for Healthcare X-ray Analysis
# Issue #2: Implement Delta tables with proper schemas using Terraform
# Using Unity Catalog (not Hive metastore)

# Bronze Layer Tables - Raw data from Kaggle

# Table: Bronze - Kaggle X-ray metadata
resource "databricks_sql_table" "bronze_kaggle_metadata" {
  name               = "kaggle_xray_metadata"
  catalog_name       = databricks_catalog.healthcare.name
  schema_name        = databricks_schema.bronze.name
  table_type         = "EXTERNAL"
  data_source_format = "DELTA"
  storage_location   = "${databricks_external_location.bronze.url}/kaggle_xray_metadata"

  comment = "Raw X-ray image metadata from Kaggle Pneumonia dataset"

  column {
    name    = "image_id"
    type    = "STRING"
    comment = "Unique identifier for X-ray image"
  }

  column {
    name    = "filename"
    type    = "STRING"
    comment = "Original filename from Kaggle"
  }

  column {
    name    = "category"
    type    = "STRING"
    comment = "Image category: NORMAL or PNEUMONIA"
  }

  column {
    name    = "dataset_split"
    type    = "STRING"
    comment = "train, test, or val split"
  }

  column {
    name    = "file_path"
    type    = "STRING"
    comment = "Path to image file in volume"
  }

  column {
    name    = "file_size_bytes"
    type    = "BIGINT"
    comment = "File size in bytes"
  }

  column {
    name    = "ingested_at"
    type    = "TIMESTAMP"
    comment = "Timestamp when data was ingested"
  }

  column {
    name    = "ingestion_batch_id"
    type    = "STRING"
    comment = "Batch ID for tracking ingestion runs"
  }

  properties = {
    "delta.enableChangeDataFeed"           = "true"
    "delta.minReaderVersion"               = "1"
    "delta.minWriterVersion"               = "2"
    "delta.writePartitionColumnsToParquet" = "true"
  }

  depends_on = [databricks_external_location.bronze]
}

# Silver Layer Tables - Cleaned and enriched data

# Table: Silver - X-ray metadata (MVP version)
resource "databricks_sql_table" "silver_xray_metadata" {
  name               = "xray_metadata"
  catalog_name       = databricks_catalog.healthcare.name
  schema_name        = databricks_schema.silver.name
  table_type         = "EXTERNAL"
  data_source_format = "DELTA"
  storage_location   = "${databricks_external_location.silver.url}/xray_metadata"

  comment = "X-ray metadata with validation and quality checks (silver layer)"

  column {
    name    = "image_id"
    type    = "STRING"
    comment = "Unique identifier for X-ray image"
  }

  column {
    name    = "label"
    type    = "INT"
    comment = "Binary label: 0=NORMAL, 1=PNEUMONIA"
  }

  column {
    name    = "label_name"
    type    = "STRING"
    comment = "Human-readable label"
  }

  column {
    name    = "dataset_split"
    type    = "STRING"
    comment = "train, test, or val split"
  }

  column {
    name    = "file_path"
    type    = "STRING"
    comment = "Validated path to image file"
  }

  column {
    name    = "image_width"
    type    = "INT"
    comment = "Image width in pixels"
  }

  column {
    name    = "image_height"
    type    = "INT"
    comment = "Image height in pixels"
  }

  column {
    name    = "quality_score"
    type    = "DOUBLE"
    comment = "Image quality score (0-1)"
  }

  column {
    name    = "processed_at"
    type    = "TIMESTAMP"
    comment = "Timestamp when data was processed"
  }

  column {
    name    = "source_batch_id"
    type    = "STRING"
    comment = "Reference to bronze batch ID"
  }

  properties = {
    "delta.enableChangeDataFeed"           = "true"
    "delta.autoOptimize.optimizeWrite"     = "true"
    "delta.autoOptimize.autoCompact"       = "true"
    "delta.writePartitionColumnsToParquet" = "true"
  }

  depends_on = [databricks_external_location.silver]
}

# Table: Silver - Image features
resource "databricks_sql_table" "silver_image_features" {
  name               = "image_features"
  catalog_name       = databricks_catalog.healthcare.name
  schema_name        = databricks_schema.silver.name
  table_type         = "EXTERNAL"
  data_source_format = "DELTA"
  storage_location   = "${databricks_external_location.silver.url}/image_features"

  comment = "Extracted features from X-ray images for ML training"

  column {
    name    = "image_id"
    type    = "STRING"
    comment = "Foreign key to silver.xray_metadata"
  }

  column {
    name    = "feature_vector"
    type    = "ARRAY<DOUBLE>"
    comment = "Extracted feature vector from CNN"
  }

  column {
    name    = "feature_extraction_model"
    type    = "STRING"
    comment = "Name/version of feature extraction model"
  }

  column {
    name    = "mean_pixel_value"
    type    = "DOUBLE"
    comment = "Mean pixel intensity"
  }

  column {
    name    = "std_pixel_value"
    type    = "DOUBLE"
    comment = "Standard deviation of pixel intensity"
  }

  column {
    name    = "extracted_at"
    type    = "TIMESTAMP"
    comment = "Timestamp when features were extracted"
  }

  properties = {
    "delta.enableChangeDataFeed"           = "true"
    "delta.writePartitionColumnsToParquet" = "true"
  }

  depends_on = [databricks_external_location.silver]
}

# Gold Layer Tables - Business-ready ML outputs

# Table: Gold - Pneumonia predictions
resource "databricks_sql_table" "gold_predictions" {
  name               = "pneumonia_predictions"
  catalog_name       = databricks_catalog.healthcare.name
  schema_name        = databricks_schema.gold.name
  table_type         = "EXTERNAL"
  data_source_format = "DELTA"
  storage_location   = "${databricks_external_location.gold.url}/pneumonia_predictions"

  comment = "ML model predictions for pneumonia classification"

  column {
    name    = "prediction_id"
    type    = "STRING"
    comment = "Unique identifier for prediction"
  }

  column {
    name    = "image_id"
    type    = "STRING"
    comment = "Foreign key to silver.xray_metadata"
  }

  column {
    name    = "predicted_label"
    type    = "INT"
    comment = "Predicted label: 0=NORMAL, 1=PNEUMONIA"
  }

  column {
    name    = "prediction_probability"
    type    = "DOUBLE"
    comment = "Probability of pneumonia (0-1)"
  }

  column {
    name    = "confidence_score"
    type    = "DOUBLE"
    comment = "Model confidence in prediction (0-1)"
  }

  column {
    name    = "true_label"
    type    = "INT"
    comment = "Actual label for validation"
  }

  column {
    name    = "is_correct"
    type    = "BOOLEAN"
    comment = "Whether prediction matches true label"
  }

  column {
    name    = "model_name"
    type    = "STRING"
    comment = "Name of ML model used"
  }

  column {
    name    = "model_version"
    type    = "STRING"
    comment = "Version of ML model"
  }

  column {
    name    = "predicted_at"
    type    = "TIMESTAMP"
    comment = "Timestamp when prediction was made"
  }

  column {
    name    = "prediction_date"
    type    = "DATE"
    comment = "Date partition for predictions"
  }

  properties = {
    "delta.enableChangeDataFeed"           = "true"
    "delta.writePartitionColumnsToParquet" = "true"
  }

  depends_on = [databricks_external_location.gold]
}

# Table: Gold - Model performance metrics
resource "databricks_sql_table" "gold_model_performance" {
  name               = "model_performance"
  catalog_name       = databricks_catalog.healthcare.name
  schema_name        = databricks_schema.gold.name
  table_type         = "EXTERNAL"
  data_source_format = "DELTA"
  storage_location   = "${databricks_external_location.gold.url}/model_performance"

  comment = "Model performance metrics and evaluation results"

  column {
    name    = "evaluation_id"
    type    = "STRING"
    comment = "Unique identifier for evaluation run"
  }

  column {
    name    = "model_name"
    type    = "STRING"
    comment = "Name of ML model"
  }

  column {
    name    = "model_version"
    type    = "STRING"
    comment = "Version of ML model"
  }

  column {
    name    = "dataset_split"
    type    = "STRING"
    comment = "Dataset used: train, test, or val"
  }

  column {
    name    = "accuracy"
    type    = "DOUBLE"
    comment = "Overall accuracy"
  }

  column {
    name    = "precision"
    type    = "DOUBLE"
    comment = "Precision score"
  }

  column {
    name    = "recall"
    type    = "DOUBLE"
    comment = "Recall score"
  }

  column {
    name    = "f1_score"
    type    = "DOUBLE"
    comment = "F1 score"
  }

  column {
    name    = "auc_roc"
    type    = "DOUBLE"
    comment = "Area under ROC curve"
  }

  column {
    name    = "confusion_matrix"
    type    = "STRING"
    comment = "Confusion matrix as JSON"
  }

  column {
    name    = "total_samples"
    type    = "BIGINT"
    comment = "Total number of samples evaluated"
  }

  column {
    name    = "evaluated_at"
    type    = "TIMESTAMP"
    comment = "Timestamp when evaluation was run"
  }

  column {
    name    = "evaluation_date"
    type    = "DATE"
    comment = "Date partition for evaluations"
  }

  properties = {
    "delta.enableChangeDataFeed"           = "true"
    "delta.writePartitionColumnsToParquet" = "true"
  }

  depends_on = [databricks_external_location.gold]
}

# Gold Layer: Prediction Feedback Table
# Stores ground truth labels from radiologists for model predictions
resource "databricks_sql_table" "gold_prediction_feedback" {
  catalog_name       = databricks_catalog.healthcare.name
  schema_name        = databricks_schema.gold.name
  name               = "prediction_feedback"
  table_type         = "EXTERNAL"
  data_source_format = "DELTA"
  storage_location   = "${databricks_external_location.gold.url}/prediction_feedback"

  comment = "Ground truth labels for model predictions - enables accuracy tracking and Champion/Challenger A/B testing"

  column {
    name     = "feedback_id"
    type     = "STRING"
    nullable = false
    comment  = "Unique feedback identifier (UUID)"
  }

  column {
    name     = "prediction_id"
    type     = "STRING"
    nullable = false
    comment  = "Links to Databricks request_id from inference table"
  }

  column {
    name     = "timestamp"
    type     = "TIMESTAMP"
    nullable = false
    comment  = "When feedback was submitted"
  }

  column {
    name     = "ground_truth"
    type     = "STRING"
    nullable = false
    comment  = "Actual diagnosis: NORMAL or PNEUMONIA"
  }

  column {
    name    = "feedback_type"
    type    = "STRING"
    comment = "Classification: true-positive, false-positive, true-negative, false-negative"
  }

  column {
    name    = "radiologist_id"
    type    = "STRING"
    comment = "ID of radiologist providing feedback"
  }

  column {
    name    = "confidence"
    type    = "STRING"
    comment = "Confidence level: confirmed, uncertain, needs_review"
  }

  column {
    name    = "feedback_source"
    type    = "STRING"
    comment = "How feedback was collected: api, radiologist, pathology, manual"
  }

  column {
    name    = "notes"
    type    = "STRING"
    comment = "Optional notes from radiologist"
  }

  properties = {
    "delta.enableChangeDataFeed"           = "true"
    "delta.autoOptimize.optimizeWrite"     = "true"
    "delta.autoOptimize.autoCompact"       = "true"
    "delta.writePartitionColumnsToParquet" = "true"
  }

  depends_on = [databricks_external_location.gold]
}

# Outputs for table references


