# Data Ingestion Notebook - Kaggle to Delta Lake
resource "databricks_notebook" "kaggle_ingestion" {
  path     = "/Shared/ingest-kaggle-xray-data"
  language = "PYTHON"
  content_base64 = base64encode(<<-EOT
    # Databricks notebook source
    # MAGIC %md
    # MAGIC # Kaggle X-ray Data Ingestion Pipeline
    # MAGIC
    # MAGIC This notebook downloads the Chest X-Ray Pneumonia dataset from Kaggle and loads it into Delta Lake tables.
    # MAGIC
    # MAGIC **Dataset**: [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
    # MAGIC
    # MAGIC **Pipeline Steps**:
    # MAGIC 1. Download data from Kaggle using API credentials
    # MAGIC 2. Extract and organize files in DBFS Volume
    # MAGIC 3. Create metadata records in Delta tables
    # MAGIC 4. Validate data quality

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ## Setup and Configuration

    # COMMAND ----------
    import os
    import json
    from datetime import datetime
    from pyspark.sql.functions import *
    from pyspark.sql.types import *

    # Get Kaggle credentials from Databricks Secrets
    kaggle_username = dbutils.secrets.get(scope="kaggle", key="username")
    kaggle_token = dbutils.secrets.get(scope="kaggle", key="token")

    # Set Kaggle environment variables
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_token

    # Configuration
    DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"
    IMAGES_PATH = "/healthcare/bronze/xray_images"
    DOWNLOAD_PATH = "/tmp/kaggle_download"
    MAX_IMAGES = 1000  # Limit to 1000 images for demo (500 per class)

    print(f"Configuration:")
    print(f"  Dataset: {DATASET_NAME}")
    print(f"  Images Path: {IMAGES_PATH}")
    print(f"  Max Images: {MAX_IMAGES}")

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ## Step 1: Install Kaggle API and Download Dataset

    # COMMAND ----------
    # Install Kaggle API
    %pip install kaggle --quiet

    # COMMAND ----------
    dbutils.library.restartPython()

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
    # MAGIC ## Step 2: Explore Downloaded Files

    # COMMAND ----------
    # List downloaded files
    dbutils.fs.ls(f"file:{DOWNLOAD_PATH}")

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ## Step 3: Organize and Copy Files to Volume (Sample)

    # COMMAND ----------
    import shutil
    from pathlib import Path

    # Create target directories
    dbutils.fs.mkdirs(f"{IMAGES_PATH}/normal")
    dbutils.fs.mkdirs(f"{IMAGES_PATH}/pneumonia")

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

            # Copy to DBFS volume
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
        f"{IMAGES_PATH}/normal",
        "NORMAL",
        max_count=500
    )

    print("Copying PNEUMONIA images...")
    pneumonia_count = copy_sample_images(
        f"{DOWNLOAD_PATH}/chest_xray/train/PNEUMONIA",
        f"{IMAGES_PATH}/pneumonia",
        "PNEUMONIA",
        max_count=500
    )

    print(f"\nTotal images copied: {normal_count + pneumonia_count}")

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ## Step 4: Create Metadata and Load into Delta Lake

    # COMMAND ----------
    from pyspark.sql.types import StructType, StructField, StringType, IntegerType, LongType, TimestampType
    from PIL import Image
    import io

    # Function to extract image metadata
    def get_image_info(file_path):
        """Extract image dimensions and size"""
        try:
            # Read image from DBFS
            image_bytes = dbutils.fs.head(file_path, maxBytes=1024*1024*10)  # Max 10MB
            img = Image.open(io.BytesIO(image_bytes.encode('latin1')))

            file_info = dbutils.fs.ls(file_path)[0]

            return {
                'width': img.width,
                'height': img.height,
                'size_bytes': file_info.size,
                'format': img.format
            }
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            return {'width': None, 'height': None, 'size_bytes': None, 'format': None}

    # COMMAND ----------
    # Create metadata records for all images
    metadata_records = []
    ingestion_time = datetime.now()

    print("Extracting metadata from NORMAL images...")
    normal_files = dbutils.fs.ls(f"{IMAGES_PATH}/normal")
    for file_info in normal_files[:500]:  # Limit to 500
        if file_info.name.endswith('.jpeg'):
            img_info = get_image_info(file_info.path)
            metadata_records.append({
                'image_id': file_info.name.replace('.jpeg', ''),
                'file_path': file_info.path,
                'label': 'NORMAL',
                'dataset_split': 'train',
                'image_width': img_info['width'],
                'image_height': img_info['height'],
                'image_size_bytes': img_info['size_bytes'],
                'image_format': img_info['format'],
                'ingestion_timestamp': ingestion_time,
                'source_dataset': DATASET_NAME
            })

    print("Extracting metadata from PNEUMONIA images...")
    pneumonia_files = dbutils.fs.ls(f"{IMAGES_PATH}/pneumonia")
    for file_info in pneumonia_files[:500]:  # Limit to 500
        if file_info.name.endswith('.jpeg'):
            img_info = get_image_info(file_info.path)
            metadata_records.append({
                'image_id': file_info.name.replace('.jpeg', ''),
                'file_path': file_info.path,
                'label': 'PNEUMONIA',
                'dataset_split': 'train',
                'image_width': img_info['width'],
                'image_height': img_info['height'],
                'image_size_bytes': img_info['size_bytes'],
                'image_format': img_info['format'],
                'ingestion_timestamp': ingestion_time,
                'source_dataset': DATASET_NAME
            })

    print(f"Total metadata records created: {len(metadata_records)}")

    # COMMAND ----------
    # Convert to DataFrame and write to Delta
    metadata_df = spark.createDataFrame(metadata_records)

    # Display sample
    display(metadata_df.limit(10))

    # COMMAND ----------
    # Write to Silver table
    metadata_df.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("silver_xray_metadata")

    print("✅ Metadata written to silver_xray_metadata")

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ## Step 5: Record Dataset Information in Bronze

    # COMMAND ----------
    # Create dataset info record
    dataset_info = spark.createDataFrame([{
        'dataset_name': DATASET_NAME,
        'dataset_version': datetime.now().strftime('%Y-%m-%d'),
        'source_url': f'https://www.kaggle.com/datasets/{DATASET_NAME}',
        'total_files': len(metadata_records),
        'total_size_bytes': sum([r['image_size_bytes'] for r in metadata_records if r['image_size_bytes']]),
        'download_timestamp': ingestion_time,
        'checksum': None  # Could add MD5/SHA256 if needed
    }])

    # Write to Bronze table
    dataset_info.write \
        .format("delta") \
        .mode("append") \
        .saveAsTable("bronze_kaggle_dataset_info")

    print("✅ Dataset info written to bronze_kaggle_dataset_info")

    # COMMAND ----------
    # MAGIC %md
    # MAGIC ## Step 6: Validation and Summary

    # COMMAND ----------
    # Query the data we just loaded
    summary_df = spark.sql("""
        SELECT
            label,
            COUNT(*) as image_count,
            AVG(image_size_bytes) as avg_size_bytes,
            AVG(image_width) as avg_width,
            AVG(image_height) as avg_height
        FROM silver_xray_metadata
        GROUP BY label
        ORDER BY label
    """)

    display(summary_df)

    # COMMAND ----------
    # Show sample records
    spark.sql("""
        SELECT *
        FROM silver_xray_metadata
        LIMIT 10
    """).display()

    # COMMAND ----------
    print("✅ Ingestion pipeline complete!")
    print(f"   - Images stored in: {IMAGES_PATH}")
    print(f"   - Metadata in: silver_xray_metadata")
    print(f"   - Dataset info in: bronze_kaggle_dataset_info")
  EOT
  )
}

# Output ingestion notebook path
output "ingestion_notebook_path" {
  value       = databricks_notebook.kaggle_ingestion.path
  description = "Path to Kaggle ingestion notebook (ready to run manually)"
}
