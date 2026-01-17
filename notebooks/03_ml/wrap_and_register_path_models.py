# Databricks notebook source
# MAGIC %md
# MAGIC # Wrap and Register Path-Based Models
# MAGIC
# MAGIC **Better Design**: Models accept Unity Catalog file paths, not image bytes
# MAGIC - Endpoint receives: `{"file_path": "/Volumes/catalog/schema/volume/image.jpg"}`
# MAGIC - Model loads image from path itself
# MAGIC - No type conversion issues (float32/float64)
# MAGIC - No network overhead sending image bytes
# MAGIC - Each model handles its own preprocessing

# COMMAND ----------
# MAGIC %pip install mlflow tensorflow==2.15.0 torch==2.1.0 pillow --quiet
# MAGIC dbutils.library.restartPython()

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

    def __init__(self, model=None):
        """Initialize with optional model"""
        self.model = model
        self.image_size = 64

    def load_context(self, context):
        """Load the Keras model from MLflow context"""
        import tensorflow as tf
        self.model = mlflow.keras.load_model(context.artifacts["model"])
        self.image_size = 64

    def predict(self, context, model_input):
        """
        Predict from file path

        Input: DataFrame with 'file_path' column
        Output: Predictions
        """
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
        """Load image from Unity Catalog volume path using Files API"""
        from databricks.sdk import WorkspaceClient
        from io import BytesIO

        # Convert dbfs:/Volumes/... to /Volumes/... for Files API
        if file_path.startswith("dbfs:"):
            file_path = file_path.replace("dbfs:", "", 1)

        # Use Files API to download from Unity Catalog volume
        w = WorkspaceClient()
        file_content = w.files.download(file_path).contents.read()

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

    def __init__(self, model=None):
        """Initialize with optional model"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        if self.model is not None:
            self.model = self.model.to(self.device)
            self.model.eval()
        self.image_size = 64

    def load_context(self, context):
        """Load the PyTorch model from MLflow context"""
        import torch
        # Load the trained model directly (don't recreate architecture)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = mlflow.pytorch.load_model(context.artifacts["model"])
        self.model = self.model.to(self.device)
        self.model.eval()

        self.image_size = 64

    def predict(self, context, model_input):
        """
        Predict from file path

        Input: DataFrame with 'file_path' column
        Output: Predictions
        """
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
        """Load image from Unity Catalog volume path using Files API"""
        from databricks.sdk import WorkspaceClient
        from io import BytesIO

        # Convert dbfs:/Volumes/... to /Volumes/... for Files API
        if file_path.startswith("dbfs:"):
            file_path = file_path.replace("dbfs:", "", 1)

        # Use Files API to download from Unity Catalog volume
        w = WorkspaceClient()
        file_content = w.files.download(file_path).contents.read()

        # Load image from bytes
        img = Image.open(BytesIO(file_content))
        img = img.convert('RGB')
        img = img.resize((self.image_size, self.image_size))

        return img

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Deploy Keras Model (Path-Based)

# COMMAND ----------
# Setup for validation
import sys
import builtins

_original_import = builtins.__import__

def _block_pyspark_import(name, *args, **kwargs):
    if 'pyspark' in name:
        raise ImportError(f"PySpark is not available in serving containers! Attempted import: {name}")
    return _original_import(name, *args, **kwargs)

mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/path-based-models")

# Load existing Keras model (byte-based version)
keras_model_name_source = "healthcare_catalog_dev.models.pneumonia_poc_classifier"
keras_model_name_remote_file = "healthcare_catalog_dev.models.pneumonia_poc_classifier_remote_file"

print(f"Loading Keras model: {keras_model_name_source} version 1")
keras_model = mlflow.keras.load_model(f"models:/{keras_model_name_source}/1")

# Create wrapper with loaded model
print("Creating path-based wrapper...")
keras_wrapper = KerasPathBasedModel(model=keras_model)

# VALIDATION: Test in serving-like environment
print("\n" + "=" * 80)
print("VALIDATING KERAS MODEL (simulating serving container - NO PYSPARK)")
print("=" * 80)

# Get a real sample path from database (has dbfs: prefix)
test_sample = spark.sql("""
    SELECT file_path FROM healthcare_catalog_dev.bronze.kaggle_xray_metadata LIMIT 1
""").collect()[0].file_path

print(f"Test file path (from DB): {test_sample}")

# Model will remove dbfs: prefix internally
print(f"Model will convert to: {test_sample.replace('dbfs:', '', 1)}")

# Block PySpark again for validation
builtins.__import__ = _block_pyspark_import

# Test predict with real data
import pandas as pd
test_df = pd.DataFrame({"file_path": [test_sample]})
try:
    test_output = keras_wrapper.predict(None, test_df)
    print(f"✓ Keras wrapper works! Output shape: {test_output.shape}")
    print(f"✓ Prediction: {test_output[0]}")
    print("=" * 80)
    print("✓✓✓ VALIDATION PASSED - Safe to deploy! ✓✓✓")
    print("=" * 80)
except Exception as e:
    print(f"✗✗✗ VALIDATION FAILED ✗✗✗")
    print(f"Error: {e}")
    print("=" * 80)
    print("DO NOT DEPLOY - Fix the code first!")
    print("=" * 80)
    raise RuntimeError(f"Model validation failed - DO NOT DEPLOY! Error: {e}")

# Restore imports for registration
builtins.__import__ = _original_import

# Create input example and infer signature
from mlflow.models import infer_signature

# Get a sample image path from bronze
sample_image = spark.sql("""
    SELECT file_path FROM healthcare_catalog_dev.bronze.kaggle_xray_metadata LIMIT 1
""").collect()[0].file_path

input_example = pd.DataFrame({"file_path": [sample_image]})

# Get sample prediction to infer signature
print("\nGenerating sample prediction for signature...")
sample_output = keras_wrapper.predict(None, input_example)
signature = infer_signature(input_example, sample_output)
print(f"Signature: {signature}")

# Register with MLflow as new model (not new version)
with mlflow.start_run(run_name="keras_remote_file"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=keras_wrapper,
        artifacts={"model": f"models:/{keras_model_name_source}/1"},
        registered_model_name=keras_model_name_remote_file,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "tensorflow==2.15.0",
            "pillow",
            "numpy<2",
            "databricks-sdk"
        ]
    )

    print(f"\n✓ Keras model registered as {keras_model_name_remote_file}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Deploy PyTorch Model (Path-Based)

# COMMAND ----------
# Load existing PyTorch model (byte-based version)
pytorch_model_name_source = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
pytorch_model_name_remote_file = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch_remote_file"

print(f"Loading PyTorch model: {pytorch_model_name_source} version 1")
pytorch_model = mlflow.pytorch.load_model(f"models:/{pytorch_model_name_source}/1")

# Create wrapper with loaded model
print("Creating path-based wrapper...")
pytorch_wrapper = PyTorchPathBasedModel(model=pytorch_model)

# VALIDATION: Test in serving-like environment
print("\n" + "=" * 80)
print("VALIDATING PYTORCH MODEL (simulating serving container - NO PYSPARK)")
print("=" * 80)

# Block PySpark for validation
builtins.__import__ = _block_pyspark_import

# Test predict with real data (reuse test_sample from Keras validation)
test_df = pd.DataFrame({"file_path": [test_sample]})
try:
    test_output = pytorch_wrapper.predict(None, test_df)
    print(f"✓ PyTorch wrapper works! Output shape: {test_output.shape}")
    print(f"✓ Prediction: {test_output[0]}")
    print("=" * 80)
    print("✓✓✓ VALIDATION PASSED - Safe to deploy! ✓✓✓")
    print("=" * 80)
except Exception as e:
    print(f"✗✗✗ VALIDATION FAILED ✗✗✗")
    print(f"Error: {e}")
    print("=" * 80)
    print("DO NOT DEPLOY - Fix the code first!")
    print("=" * 80)
    raise RuntimeError(f"PyTorch model validation failed - DO NOT DEPLOY! Error: {e}")

# Restore imports for registration
builtins.__import__ = _original_import

# Create input example and infer signature (reuse sample_image from Keras section)
input_example = pd.DataFrame({"file_path": [sample_image]})

# Get sample prediction to infer signature
print("\nGenerating sample prediction for signature...")
sample_output = pytorch_wrapper.predict(None, input_example)
signature = infer_signature(input_example, sample_output)
print(f"Signature: {signature}")

# Register with MLflow as new model (not new version)
with mlflow.start_run(run_name="pytorch_remote_file"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=pytorch_wrapper,
        artifacts={"model": f"models:/{pytorch_model_name_source}/1"},
        registered_model_name=pytorch_model_name_remote_file,
        signature=signature,
        input_example=input_example,
        pip_requirements=[
            "torch==2.1.0",
            "pillow",
            "numpy<2",
            "databricks-sdk"
        ]
    )

    print(f"\n✓ PyTorch model registered as {pytorch_model_name_remote_file}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **Validation Strategy**:
# MAGIC - Blocked PySpark imports to simulate serving container environment
# MAGIC - Tested both models with real data BEFORE registration
# MAGIC - If validation passes, deployment will work
# MAGIC - If validation fails, fix code immediately (no 10-12 min wasted)
# MAGIC
# MAGIC **What Changed**:
# MAGIC - Models now accept `file_path` instead of image bytes
# MAGIC - Each model loads and preprocesses images itself
# MAGIC - No PySpark dependency (uses direct filesystem access)
# MAGIC - No type conversion issues (float32/float64)
# MAGIC - Much more efficient (no sending image bytes over network)
