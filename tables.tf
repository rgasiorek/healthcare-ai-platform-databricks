# SQL Notebook to create Delta tables with schema definitions
resource "databricks_notebook" "create_tables" {
  path     = "/Shared/setup-delta-tables"
  language = "SQL"
  content_base64 = base64encode(<<-EOT
    -- Databricks notebook source
    -- MAGIC %md
    -- MAGIC # Healthcare Delta Lake Table Definitions (Hive Metastore)
    -- MAGIC
    -- MAGIC This notebook creates Delta Lake tables using the default Hive metastore.
    -- MAGIC
    -- MAGIC **Note**: Using Hive metastore instead of Unity Catalog (no AWS S3 setup required)
    -- MAGIC
    -- MAGIC **Table Organization:**
    -- MAGIC - Bronze tables: Raw Kaggle data (prefix: bronze_)
    -- MAGIC - Silver tables: Cleaned metadata (prefix: silver_)
    -- MAGIC - Gold tables: ML results and aggregations (prefix: gold_)

    -- COMMAND ----------
    -- MAGIC %md
    -- MAGIC ## Bronze Layer: Kaggle Dataset Metadata

    -- COMMAND ----------
    CREATE TABLE IF NOT EXISTS bronze_kaggle_dataset_info (
      dataset_name STRING COMMENT 'Name of the Kaggle dataset',
      dataset_version STRING COMMENT 'Version or download timestamp',
      source_url STRING COMMENT 'Kaggle dataset URL',
      total_files INT COMMENT 'Total number of files downloaded',
      total_size_bytes BIGINT COMMENT 'Total size in bytes',
      download_timestamp TIMESTAMP COMMENT 'When the dataset was downloaded',
      checksum STRING COMMENT 'Dataset checksum for validation'
    )
    USING DELTA
    COMMENT 'Metadata about ingested Kaggle datasets'
    TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true',
      'quality' = 'bronze'
    );

    -- COMMAND ----------
    -- MAGIC %md
    -- MAGIC ## Silver Layer: X-ray Image Metadata

    -- COMMAND ----------
    CREATE TABLE IF NOT EXISTS silver_xray_metadata (
      image_id STRING COMMENT 'Unique identifier for each X-ray image',
      file_path STRING COMMENT 'Path to image file in DBFS volume',
      label STRING COMMENT 'Classification label: NORMAL or PNEUMONIA',
      dataset_split STRING COMMENT 'train, test, or validation',
      image_width INT COMMENT 'Image width in pixels',
      image_height INT COMMENT 'Image height in pixels',
      image_size_bytes BIGINT COMMENT 'File size in bytes',
      image_format STRING COMMENT 'File format (e.g., JPEG, PNG)',
      ingestion_timestamp TIMESTAMP COMMENT 'When the record was created',
      source_dataset STRING COMMENT 'Source Kaggle dataset name'
    )
    USING DELTA
    PARTITIONED BY (label, dataset_split)
    COMMENT 'Metadata for all X-ray images with labels and file information'
    TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true',
      'delta.autoOptimize.optimizeWrite' = 'true',
      'delta.autoOptimize.autoCompact' = 'true',
      'quality' = 'silver'
    );

    -- COMMAND ----------
    -- MAGIC %md
    -- MAGIC ## Silver Layer: Image Features (for ML)

    -- COMMAND ----------
    CREATE TABLE IF NOT EXISTS silver_image_features (
      image_id STRING COMMENT 'Links to xray_metadata.image_id',
      feature_vector ARRAY<DOUBLE> COMMENT 'Extracted feature vector from pre-trained model',
      feature_extraction_model STRING COMMENT 'Model used for feature extraction',
      extraction_timestamp TIMESTAMP COMMENT 'When features were extracted',
      image_mean_intensity DOUBLE COMMENT 'Mean pixel intensity',
      image_std_intensity DOUBLE COMMENT 'Standard deviation of pixel intensity',
      contrast_ratio DOUBLE COMMENT 'Image contrast metric'
    )
    USING DELTA
    COMMENT 'Extracted features from X-ray images for ML modeling'
    TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true',
      'quality' = 'silver'
    );

    -- COMMAND ----------
    -- MAGIC %md
    -- MAGIC ## Gold Layer: ML Model Predictions

    -- COMMAND ----------
    CREATE TABLE IF NOT EXISTS gold_pneumonia_predictions (
      prediction_id STRING COMMENT 'Unique prediction identifier',
      image_id STRING COMMENT 'Links to xray_metadata.image_id',
      actual_label STRING COMMENT 'Ground truth label',
      predicted_label STRING COMMENT 'Model prediction: NORMAL or PNEUMONIA',
      prediction_probability DOUBLE COMMENT 'Confidence score (0-1)',
      model_name STRING COMMENT 'Name of the ML model used',
      model_version STRING COMMENT 'Version of the model',
      prediction_timestamp TIMESTAMP COMMENT 'When prediction was made',
      is_correct BOOLEAN COMMENT 'Whether prediction matches actual label'
    )
    USING DELTA
    PARTITIONED BY (predicted_label)
    COMMENT 'ML model predictions with confidence scores and validation'
    TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true',
      'quality' = 'gold'
    );

    -- COMMAND ----------
    -- MAGIC %md
    -- MAGIC ## Gold Layer: Model Performance Metrics

    -- COMMAND ----------
    CREATE TABLE IF NOT EXISTS gold_model_performance (
      model_name STRING COMMENT 'Name of the ML model',
      model_version STRING COMMENT 'Version of the model',
      evaluation_date DATE COMMENT 'When the model was evaluated',
      accuracy DOUBLE COMMENT 'Overall accuracy',
      precision DOUBLE COMMENT 'Precision score',
      recall DOUBLE COMMENT 'Recall score',
      f1_score DOUBLE COMMENT 'F1 score',
      auc_roc DOUBLE COMMENT 'Area under ROC curve',
      true_positives INT COMMENT 'Count of true positives',
      true_negatives INT COMMENT 'Count of true negatives',
      false_positives INT COMMENT 'Count of false positives',
      false_negatives INT COMMENT 'Count of false negatives',
      total_predictions INT COMMENT 'Total number of predictions evaluated'
    )
    USING DELTA
    COMMENT 'Performance metrics for ML models over time'
    TBLPROPERTIES (
      'delta.enableChangeDataFeed' = 'true',
      'quality' = 'gold'
    );

    -- COMMAND ----------
    -- MAGIC %md
    -- MAGIC ## Gold Layer: Dataset Summary View

    -- COMMAND ----------
    CREATE OR REPLACE VIEW gold_dataset_summary AS
    SELECT
      label,
      dataset_split,
      COUNT(*) as image_count,
      AVG(image_size_bytes) as avg_size_bytes,
      MIN(ingestion_timestamp) as first_ingested,
      MAX(ingestion_timestamp) as last_ingested
    FROM silver_xray_metadata
    GROUP BY label, dataset_split
    COMMENT 'Summary statistics of the X-ray dataset';

    -- COMMAND ----------
    -- MAGIC %md
    -- MAGIC ## Verify Table Creation

    -- COMMAND ----------
    -- Show all tables in default database
    SHOW TABLES;

    -- COMMAND ----------
    -- Filter to show only our healthcare tables
    SHOW TABLES LIKE '*xray*';

    -- COMMAND ----------
    -- Display table properties
    DESCRIBE EXTENDED silver_xray_metadata;
  EOT
  )
}

# Output table setup notebook
output "tables_notebook_path" {
  value       = databricks_notebook.create_tables.path
  description = "Path to Delta tables setup notebook"
}
