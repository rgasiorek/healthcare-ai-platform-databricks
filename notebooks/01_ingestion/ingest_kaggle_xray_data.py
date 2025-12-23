# Databricks notebook source
# MAGIC %md
# MAGIC # Kaggle X-ray Data Ingestion Pipeline
# MAGIC
# MAGIC This notebook downloads the Chest X-Ray Pneumonia dataset from Kaggle and loads it into Delta Lake tables.
# MAGIC
# MAGIC **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
# MAGIC
# MAGIC **Pipeline Steps**:
# MAGIC 1. Install Kaggle API and restart Python
# MAGIC 2. Set credentials and download data from Kaggle
# MAGIC 3. Extract and organize files in Unity Catalog Volume
# MAGIC 4. Create metadata records in Bronze Delta table
# MAGIC 5. Validate data quality

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Install Kaggle API

# COMMAND ----------
# Install Kaggle API
%pip install kaggle --quiet

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Restart Python (Required after pip install)

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Setup Configuration and Kaggle Credentials

# COMMAND ----------
import os
import json
from datetime import datetime
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Configuration
DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"
CATALOG = "healthcare_catalog_dev"
SCHEMA_BRONZE = "bronze"
SCHEMA_SILVER = "silver"
VOLUME_PATH = "/Volumes/healthcare_catalog_dev/bronze/xray_images"
DOWNLOAD_PATH = "/tmp/kaggle_download"
BATCH_ID = datetime.now().strftime('%Y%m%d_%H%M%S')
MAX_IMAGES = 1000  # Limit to 1000 images for demo (500 per class)

# Get Kaggle credentials from Databricks Secrets (AFTER restart!)
kaggle_username = dbutils.secrets.get(scope="kaggle", key="username")
kaggle_token = dbutils.secrets.get(scope="kaggle", key="token")

# Set Kaggle environment variables
os.environ['KAGGLE_USERNAME'] = kaggle_username
os.environ['KAGGLE_KEY'] = kaggle_token

print(f"Configuration:")
print(f"  Dataset: {DATASET_NAME}")
print(f"  Catalog: {CATALOG}")
print(f"  Volume Path: {VOLUME_PATH}")
print(f"  Batch ID: {BATCH_ID}")
print(f"  Max Images: {MAX_IMAGES}")
print(f"  Kaggle User: {kaggle_username}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Download Dataset from Kaggle

# COMMAND ----------
import kaggle
from kaggle.api.kaggle_api_extended import KaggleApi

# Authenticate and download
api = KaggleApi()
api.authenticate()

print(f"Downloading dataset: {DATASET_NAME}")
print(f"This may take several minutes...")

# Download to temp location
api.dataset_download_files(
    DATASET_NAME,
    path=DOWNLOAD_PATH,
    unzip=True
)

print("Download complete!")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Explore Downloaded Files

# COMMAND ----------
# List downloaded files
dbutils.fs.ls(f"file:{DOWNLOAD_PATH}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Organize and Copy Files to Unity Catalog Volume (Sample)

# COMMAND ----------
import shutil
from pathlib import Path

# Create target directories in Unity Catalog volume
dbutils.fs.mkdirs(f"{VOLUME_PATH}/normal")
dbutils.fs.mkdirs(f"{VOLUME_PATH}/pneumonia")

# Function to copy sample images
def copy_sample_images(source_dir, target_dir, label, max_count=500):
    """Copy a sample of images from source to target directory"""
    source_path = Path(source_dir)
    copied = 0

    if not source_path.exists():
        print(f"Warning: {source_dir} does not exist")
        return 0

    for image_file in source_path.glob("*.jpeg"):
        if copied >= max_count:
            break

        # Copy to Unity Catalog volume
        target_file = f"{target_dir}/{image_file.name}"
        dbutils.fs.cp(f"file:{str(image_file)}", target_file)
        copied += 1

        if copied % 100 == 0:
            print(f"  Copied {copied} {label} images...")

    print(f"Total {label} images copied: {copied}")
    return copied

# Copy normal and pneumonia images (sample)
print("Copying NORMAL images...")
normal_count = copy_sample_images(
    f"{DOWNLOAD_PATH}/chest_xray/train/NORMAL",
    f"{VOLUME_PATH}/normal",
    "NORMAL",
    max_count=500
)

print("Copying PNEUMONIA images...")
pneumonia_count = copy_sample_images(
    f"{DOWNLOAD_PATH}/chest_xray/train/PNEUMONIA",
    f"{VOLUME_PATH}/pneumonia",
    "PNEUMONIA",
    max_count=500
)

print(f"\nTotal images copied: {normal_count + pneumonia_count}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Create Metadata and Load into Bronze Delta Lake Table

# COMMAND ----------
from pyspark.sql.types import StructType, StructField, StringType, LongType, TimestampType

# Create metadata records for Bronze table
# Schema: image_id, filename, category, dataset_split, file_path, file_size_bytes, ingested_at, ingestion_batch_id
bronze_records = []
ingestion_time = datetime.now()

print("Creating metadata for NORMAL images...")
normal_files = dbutils.fs.ls(f"{VOLUME_PATH}/normal")
for file_info in normal_files[:500]:  # Limit to 500
    if file_info.name.endswith('.jpeg'):
        bronze_records.append({
            'image_id': file_info.name.replace('.jpeg', ''),
            'filename': file_info.name,
            'category': 'NORMAL',
            'dataset_split': 'train',
            'file_path': file_info.path,
            'file_size_bytes': file_info.size,
            'ingested_at': ingestion_time,
            'ingestion_batch_id': BATCH_ID
        })

print("Creating metadata for PNEUMONIA images...")
pneumonia_files = dbutils.fs.ls(f"{VOLUME_PATH}/pneumonia")
for file_info in pneumonia_files[:500]:  # Limit to 500
    if file_info.name.endswith('.jpeg'):
        bronze_records.append({
            'image_id': file_info.name.replace('.jpeg', ''),
            'filename': file_info.name,
            'category': 'PNEUMONIA',
            'dataset_split': 'train',
            'file_path': file_info.path,
            'file_size_bytes': file_info.size,
            'ingested_at': ingestion_time,
            'ingestion_batch_id': BATCH_ID
        })

print(f"Total bronze records created: {len(bronze_records)}")

# COMMAND ----------
# Convert to DataFrame
bronze_df = spark.createDataFrame(bronze_records)

# Display sample
display(bronze_df.limit(10))

# COMMAND ----------
# Write to Bronze table (using Unity Catalog three-level namespace)
bronze_table = f"{CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata"

bronze_df.write \
    .format("delta") \
    .mode("append") \
    .saveAsTable(bronze_table)

print(f"✅ Metadata written to {bronze_table}")
print(f"   Records: {len(bronze_records)}")
print(f"   Batch ID: {BATCH_ID}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 8: Validation and Summary

# COMMAND ----------
# Query the bronze data we just loaded
summary_df = spark.sql(f"""
    SELECT
        category,
        dataset_split,
        COUNT(*) as image_count,
        SUM(file_size_bytes) as total_size_bytes,
        AVG(file_size_bytes) as avg_size_bytes,
        MIN(ingested_at) as first_ingested,
        MAX(ingested_at) as last_ingested
    FROM {CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata
    WHERE ingestion_batch_id = '{BATCH_ID}'
    GROUP BY category, dataset_split
    ORDER BY category, dataset_split
""")

print("📊 Ingestion Summary:")
display(summary_df)

# COMMAND ----------
# Show sample records from Bronze table
print("Sample Bronze Records:")
spark.sql(f"""
    SELECT
        image_id,
        filename,
        category,
        dataset_split,
        file_size_bytes,
        ingested_at
    FROM {CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata
    WHERE ingestion_batch_id = '{BATCH_ID}'
    LIMIT 10
""").display()

# COMMAND ----------
# Count total records
total_count = spark.sql(f"""
    SELECT COUNT(*) as total
    FROM {CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata
    WHERE ingestion_batch_id = '{BATCH_ID}'
""").collect()[0]['total']

print("✅ Ingestion pipeline complete!")
print(f"   - Batch ID: {BATCH_ID}")
print(f"   - Images stored in: {VOLUME_PATH}")
print(f"   - Metadata in: {CATALOG}.{SCHEMA_BRONZE}.kaggle_xray_metadata")
print(f"   - Total records ingested: {total_count}")
print(f"\nNext steps:")
print(f"   1. Transform bronze → silver data (with image processing)")
print(f"   2. Create gold layer predictions (ML model)")
