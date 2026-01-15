# Databricks notebook source
# MAGIC %md
# MAGIC # Fix PyTorch Model - Add Float32 Conversion Wrapper
# MAGIC
# MAGIC **Problem**: PyTorch model fails when receiving float64 inputs from endpoint
# MAGIC **Solution**: Re-register model with custom wrapper that converts to float32
# MAGIC **Note**: Model weights unchanged - only wrapping prediction function

# COMMAND ----------
import mlflow
import torch
import numpy as np
from mlflow.pyfunc import PythonModel

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Load Existing PyTorch Model

# COMMAND ----------
# Load the existing model from Unity Catalog
model_name = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
model_version = "1"

print(f"Loading model: {model_name} version {model_version}")
pytorch_model = mlflow.pytorch.load_model(f"models:/{model_name}/{model_version}")
print("✓ Model loaded successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Create Custom Wrapper with Float32 Conversion

# COMMAND ----------
class Float32ConverterWrapper(PythonModel):
    """
    Custom MLflow PyFunc wrapper that converts inputs to float32
    Fixes TypeError: Input type (double) and bias type (float) should be the same
    """

    def __init__(self, pytorch_model):
        self.model = pytorch_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def predict(self, context, model_input):
        """
        Predict with automatic float32 conversion

        Args:
            context: MLflow context (unused)
            model_input: Input data (numpy array or pandas DataFrame)

        Returns:
            Predictions as numpy array
        """
        # Convert input to numpy if it's a DataFrame
        if hasattr(model_input, 'values'):
            input_array = model_input.values
        else:
            input_array = model_input

        # Convert to float32 (fixes the PyTorch type error)
        input_array = input_array.astype(np.float32)

        # Convert to PyTorch tensor
        # Input shape: (batch, 64, 64, 3) -> need (batch, 3, 64, 64) for PyTorch
        input_tensor = torch.FloatTensor(input_array).permute(0, 3, 1, 2).to(self.device)

        # Get predictions
        with torch.no_grad():
            output = self.model(input_tensor)

        # Convert back to numpy
        predictions = output.cpu().numpy()

        return predictions

# Create wrapped model instance
wrapped_model = Float32ConverterWrapper(pytorch_model)
print("✓ Wrapper created successfully")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Test Wrapper with Sample Input

# COMMAND ----------
# Create test input (simulating what the endpoint sends)
test_input = np.random.rand(1, 64, 64, 3).astype(np.float64)  # float64 like endpoint sends
print(f"Test input dtype: {test_input.dtype} (float64 - this causes the error)")

# Test prediction with wrapper
try:
    test_output = wrapped_model.predict(None, test_input)
    print(f"✓ Prediction successful!")
    print(f"  Output shape: {test_output.shape}")
    print(f"  Output value: {test_output[0][0]:.4f}")
except Exception as e:
    print(f"✗ Prediction failed: {e}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Re-register Model with Wrapper

# COMMAND ----------
# Set MLflow tracking
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/pytorch-model-fix")

# Create sample input for signature
sample_input = np.random.rand(1, 64, 64, 3)
sample_output = wrapped_model.predict(None, sample_input)

# Infer signature
from mlflow.models import infer_signature
signature = infer_signature(sample_input, sample_output)

# Log the wrapped model (this creates version 2)
with mlflow.start_run(run_name="pytorch_model_with_float32_fix"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=wrapped_model,
        registered_model_name=model_name,
        signature=signature,
        pip_requirements=[
            "torch==2.1.0",
            "numpy"
        ]
    )

    print(f"✓ Model re-registered as version 2 with float32 conversion wrapper")
    print(f"  Model name: {model_name}")
    print(f"  New version will be created automatically")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Update Terraform to Use Version 2
# MAGIC
# MAGIC **Manual Step Required**:
# MAGIC Update `terraform/databricks/endpoints.tf`:
# MAGIC
# MAGIC ```hcl
# MAGIC served_entities {
# MAGIC   entity_name    = "healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch"
# MAGIC   entity_version = "2"  # Changed from "1" to "2"
# MAGIC   ...
# MAGIC }
# MAGIC ```
# MAGIC
# MAGIC Then run `terraform apply` to update the endpoint.

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC **What we did**:
# MAGIC - Loaded existing PyTorch model (version 1)
# MAGIC - Wrapped it with Float32ConverterWrapper that converts inputs to float32
# MAGIC - Re-registered as version 2 (model weights unchanged)
# MAGIC
# MAGIC **Next steps**:
# MAGIC 1. Update Terraform config to use version 2
# MAGIC 2. Run `terraform apply`
# MAGIC 3. Test endpoint - PyTorch model should now accept float64 inputs
# MAGIC 4. Dashboard will show both models in comparison
