# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Path-Based Model Endpoints
# MAGIC
# MAGIC **Better Design**: Models accept Unity Catalog file paths, not image bytes
# MAGIC - Endpoint receives: `{"file_path": "/Volumes/catalog/schema/volume/image.jpg"}`
# MAGIC - Model loads image from path itself
# MAGIC - No type conversion issues (float32/float64)
# MAGIC - No network overhead sending image bytes
# MAGIC - Each model handles its own preprocessing

# COMMAND ----------
import mlflow
from mlflow.pyfunc import PythonModel
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from io import BytesIO

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Define Keras Model Wrapper (Path-Based)

# COMMAND ----------
class KerasPathBasedModel(PythonModel):
    """
    Keras model that accepts file paths instead of image bytes
    """

    def load_context(self, context):
        """Load the Keras model"""
        import tensorflow as tf
        self.model = mlflow.keras.load_model(context.artifacts["model"])
        self.image_size = 64

    def predict(self, context, model_input):
        """
        Predict from file path

        Input: DataFrame with 'file_path' column
        Output: Predictions
        """
        from pyspark.sql import SparkSession

        predictions = []

        # Get file paths from input
        if hasattr(model_input, 'values'):
            # DataFrame input
            file_paths = model_input['file_path'].values if 'file_path' in model_input.columns else model_input.iloc[:, 0].values
        else:
            # List or array input
            file_paths = model_input

        for file_path in file_paths:
            # Load image from Unity Catalog volume
            img = self._load_image_from_path(file_path)

            # Preprocess for Keras (TensorFlow/Keras handles types automatically)
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            # Predict
            pred = self.model.predict(img_array, verbose=0)
            predictions.append(pred[0])

        return np.array(predictions)

    def _load_image_from_path(self, file_path):
        """Load image from Unity Catalog volume path"""
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        # Unity Catalog volumes are accessed via Spark
        if file_path.startswith("dbfs:"):
            volume_path = file_path
        else:
            volume_path = file_path

        # Read file using Spark binaryFiles
        binary_df = spark.read.format("binaryFile").load(volume_path)
        file_content = binary_df.select("content").collect()[0][0]

        # Load image from bytes
        img = Image.open(BytesIO(file_content))
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size))

        return img

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Define PyTorch Model Architecture (Same as Training)

# COMMAND ----------
class SimpleCNN(nn.Module):
    """Simple CNN for pneumonia classification (matches training notebook)"""

    def __init__(self, image_size=64):
        super(SimpleCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size after conv and pooling
        conv_output_size = (image_size // 4) * (image_size // 4) * 64

        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, 1)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Define PyTorch Model Wrapper (Path-Based)

# COMMAND ----------
class PyTorchPathBasedModel(PythonModel):
    """
    PyTorch model that accepts file paths instead of image bytes
    Handles its own preprocessing - no type conversion issues
    """

    def load_context(self, context):
        """Load the PyTorch model"""
        # Load model architecture
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SimpleCNN(image_size=64).to(self.device)

        # Load trained weights
        import torch
        model_state = mlflow.pytorch.load_model(context.artifacts["model"])
        self.model.load_state_dict(model_state.state_dict())
        self.model.eval()

        self.image_size = 64

    def predict(self, context, model_input):
        """
        Predict from file path

        Input: DataFrame with 'file_path' column
        Output: Predictions
        """
        from pyspark.sql import SparkSession

        predictions = []

        # Get file paths from input
        if hasattr(model_input, 'values'):
            # DataFrame input
            file_paths = model_input['file_path'].values if 'file_path' in model_input.columns else model_input.iloc[:, 0].values
        else:
            # List or array input
            file_paths = model_input

        for file_path in file_paths:
            # Load image from Unity Catalog volume
            img = self._load_image_from_path(file_path)

            # Preprocess for PyTorch (handles its own types - no conversion issues!)
            img_array = np.array(img) / 255.0
            img_tensor = torch.FloatTensor(img_array).permute(2, 0, 1)  # (H,W,C) -> (C,H,W)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)  # Add batch dimension

            # Predict
            with torch.no_grad():
                pred = self.model(img_tensor).cpu().numpy()

            predictions.append(pred[0])

        return np.array(predictions)

    def _load_image_from_path(self, file_path):
        """Load image from Unity Catalog volume path"""
        from pyspark.sql import SparkSession
        spark = SparkSession.builder.getOrCreate()

        # Unity Catalog volumes are accessed via Spark
        if file_path.startswith("dbfs:"):
            volume_path = file_path
        else:
            volume_path = file_path

        # Read file using Spark binaryFiles
        binary_df = spark.read.format("binaryFile").load(volume_path)
        file_content = binary_df.select("content").collect()[0][0]

        # Load image from bytes
        img = Image.open(BytesIO(file_content))
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size))

        return img

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Deploy Keras Model (Path-Based)

# COMMAND ----------
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/path-based-models")

# Load existing Keras model
keras_model_name = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
print(f"Loading Keras model: {keras_model_name} version 1")
keras_model = mlflow.keras.load_model(f"models:/{keras_model_name}/1")

# Create wrapper
print("Creating path-based wrapper...")
keras_wrapper = KerasPathBasedModel()

# Test with sample path
print("\nTesting Keras wrapper with sample path...")
sample_path = "dbfs:/Volumes/healthcare_catalog_dev/bronze/xray_images/NORMAL/IM-0001-0001.jpeg"
# Note: Can't test here without Spark session - will test after deployment

# Create input example for signature
import pandas as pd
input_example = pd.DataFrame({"file_path": [sample_path]})

# Register with MLflow
with mlflow.start_run(run_name="keras_path_based"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=keras_wrapper,
        artifacts={"model": f"models:/{keras_model_name}/1"},
        registered_model_name=keras_model_name,
        input_example=input_example,
        pip_requirements=[
            "tensorflow==2.15.0",
            "pillow"
        ]
    )

    print(f"\n✓ Keras model registered as new version (path-based)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Deploy PyTorch Model (Path-Based)

# COMMAND ----------
# Load existing PyTorch model
pytorch_model_name = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
print(f"Loading PyTorch model: {pytorch_model_name} version 1")
pytorch_model = mlflow.pytorch.load_model(f"models:/{pytorch_model_name}/1")

# Create wrapper
print("Creating path-based wrapper...")
pytorch_wrapper = PyTorchPathBasedModel()

# Create input example for signature
input_example = pd.DataFrame({"file_path": [sample_path]})

# Register with MLflow
with mlflow.start_run(run_name="pytorch_path_based"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=pytorch_wrapper,
        artifacts={"model": f"models:/{pytorch_model_name}/1"},
        registered_model_name=pytorch_model_name,
        input_example=input_example,
        pip_requirements=[
            "torch==2.1.0",
            "pillow"
        ]
    )

    print(f"\n✓ PyTorch model registered as new version (path-based)")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Update Terraform and Demo
# MAGIC
# MAGIC **Terraform Update** (`terraform/databricks/endpoints.tf`):
# MAGIC ```hcl
# MAGIC served_entities {
# MAGIC   entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
# MAGIC   entity_version = "2"  # Updated to path-based version
# MAGIC   ...
# MAGIC }
# MAGIC served_entities {
# MAGIC   entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
# MAGIC   entity_version = "2"  # Updated to path-based version
# MAGIC   ...
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC **Demo Update** (`notebooks/05_demo/end_to_end_demo.py`):
# MAGIC ```python
# MAGIC # OLD: Send image bytes
# MAGIC # payload = {"inputs": [img_array.tolist()]}
# MAGIC
# MAGIC # NEW: Send file path
# MAGIC payload = {"dataframe_records": [{"file_path": img.file_path}]}
# MAGIC ```
# MAGIC
# MAGIC Then run `terraform apply`

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What Changed**:
# MAGIC - Models now accept `file_path` instead of image bytes
# MAGIC - Each model loads and preprocesses images itself
# MAGIC - No type conversion issues (float32/float64)
# MAGIC - Much more efficient (no sending image bytes over network)
# MAGIC
# MAGIC **Benefits**:
# MAGIC - PyTorch model will work (handles its own preprocessing)
# MAGIC - No REST payload size limits
# MAGIC - Models control their own preprocessing pipeline
# MAGIC - Cleaner separation of concerns
# MAGIC
# MAGIC **Next Steps**:
# MAGIC 1. Run this notebook to register path-based models (versions 2)
# MAGIC 2. Update Terraform to use version 2
# MAGIC 3. Update demo to send paths not bytes
# MAGIC 4. Run `terraform apply`
# MAGIC 5. Dashboard will show both models compared
