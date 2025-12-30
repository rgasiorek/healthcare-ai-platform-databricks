# Databricks notebook source
# MAGIC %md
# MAGIC # Pneumonia Classification POC - Train Model
# MAGIC
# MAGIC **Simplified approach - just TensorFlow, no Spark, no scikit-learn!**
# MAGIC
# MAGIC This notebook trains a simple CNN classifier on sample X-ray data:
# MAGIC 1. Load 100 images directly from filesystem (50 NORMAL, 50 PNEUMONIA)
# MAGIC 2. Preprocess and split into train/val/test sets
# MAGIC 3. Train simple CNN model (~5-10 min on CPU)
# MAGIC 4. Register model in MLflow Model Registry
# MAGIC
# MAGIC **Goal**: Prove training workflow works. Accept any model accuracy (50-80% is fine for POC).
# MAGIC
# MAGIC **Libraries used**: Only TensorFlow + Pillow (no unnecessary frameworks!)
# MAGIC
# MAGIC **Next Steps**: Deploy registered model as serving endpoint (see `/Shared/deploy-serving-endpoint`)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------
# Install required libraries (TensorFlow, Pillow, and MLflow)
%pip install tensorflow==2.15.0 Pillow mlflow --quiet

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Restart Python

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Configuration and Imports

# COMMAND ----------
import os
import numpy as np
from PIL import Image

# TensorFlow/Keras (only framework we need!)
import tensorflow as tf
from tensorflow.keras import layers, models

# MLflow
import mlflow
import mlflow.keras

# Configuration
VOLUME_PATH = "/Volumes/healthcare_catalog_dev/bronze/xray_images"  # Unity Catalog volume (no /dbfs prefix!)
SAMPLE_SIZE_PER_CLASS = 50  # 50 NORMAL + 50 PNEUMONIA = 100 total
IMAGE_SIZE = 64    # 64x64 for fast training
TRAIN_SPLIT = 0.70  # 70% train
VAL_SPLIT = 0.15    # 15% validation
TEST_SPLIT = 0.15   # 15% test

print(f"Configuration:")
print(f"  Volume: {VOLUME_PATH}")
print(f"  Sample Size: {SAMPLE_SIZE_PER_CLASS * 2} images (50 per class)")
print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Splits: {TRAIN_SPLIT:.0%} train, {VAL_SPLIT:.0%} val, {TEST_SPLIT:.0%} test")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Load Images Directly from Filesystem
# MAGIC
# MAGIC No Spark needed! Just read image files directly from Unity Catalog volume.

# COMMAND ----------
# List image files directly from volume (no Spark/SQL needed!)
normal_path = f"{VOLUME_PATH}/normal"
pneumonia_path = f"{VOLUME_PATH}/pneumonia"

normal_files = [f for f in os.listdir(normal_path) if f.endswith('.jpeg')][:SAMPLE_SIZE_PER_CLASS]
pneumonia_files = [f for f in os.listdir(pneumonia_path) if f.endswith('.jpeg')][:SAMPLE_SIZE_PER_CLASS]

print(f"Found images:")
print(f"  NORMAL: {len(normal_files)} files")
print(f"  PNEUMONIA: {len(pneumonia_files)} files")
print(f"  Total: {len(normal_files) + len(pneumonia_files)} files")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Load and Preprocess Images

# COMMAND ----------
def load_and_preprocess_image(file_path, image_size=64):
    """Load image from file path and preprocess for model input"""
    try:
        # Load image using PIL
        img = Image.open(file_path)

        # Convert to RGB (in case of grayscale)
        img = img.convert('RGB')

        # Resize to target size
        img = img.resize((image_size, image_size))

        # Convert to numpy array and normalize to [0, 1]
        img_array = np.array(img) / 255.0

        return img_array
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

# Load all images and labels
X = []
y = []

print(f"Loading NORMAL images...")
for i, filename in enumerate(normal_files):
    img_array = load_and_preprocess_image(f"{normal_path}/{filename}", IMAGE_SIZE)
    if img_array is not None:
        X.append(img_array)
        y.append(0)  # 0 = NORMAL

    if (i + 1) % 20 == 0:
        print(f"  Loaded {i + 1}/{len(normal_files)}...")

print(f"Loading PNEUMONIA images...")
for i, filename in enumerate(pneumonia_files):
    img_array = load_and_preprocess_image(f"{pneumonia_path}/{filename}", IMAGE_SIZE)
    if img_array is not None:
        X.append(img_array)
        y.append(1)  # 1 = PNEUMONIA

    if (i + 1) % 20 == 0:
        print(f"  Loaded {i + 1}/{len(pneumonia_files)}...")

# Convert to numpy arrays
X = np.array(X)
y = np.array(y)

print(f"\nData loaded successfully:")
print(f"  X shape: {X.shape}")
print(f"  y shape: {y.shape}")
print(f"  NORMAL: {np.sum(y == 0)} images")
print(f"  PNEUMONIA: {np.sum(y == 1)} images")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Create Train/Validation/Test Splits

# COMMAND ----------
# Calculate split sizes
total_samples = len(X)
train_size = int(total_samples * TRAIN_SPLIT)
val_size = int(total_samples * VAL_SPLIT)
test_size = total_samples - train_size - val_size

# Shuffle indices
np.random.seed(42)
indices = np.random.permutation(total_samples)

# Split data
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:]

X_train, y_train = X[train_indices], y[train_indices]
X_val, y_val = X[val_indices], y[val_indices]
X_test, y_test = X[test_indices], y[test_indices]

print(f"Data splits:")
print(f"  Train: {len(X_train)} samples ({len(X_train)/total_samples:.1%})")
print(f"  Val:   {len(X_val)} samples ({len(X_val)/total_samples:.1%})")
print(f"  Test:  {len(X_test)} samples ({len(X_test)/total_samples:.1%})")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Build Simple CNN Model

# COMMAND ----------
# Build a simple CNN for binary classification
model = models.Sequential([
    # First convolutional block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    layers.MaxPooling2D((2, 2)),

    # Second convolutional block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # Flatten and dense layers
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')  # Binary classification
])

# Compile model with built-in Keras metrics (no scikit-learn needed!)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[
        'accuracy',
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc')
    ]
)

# Display model architecture
model.summary()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 8: Configure MLflow and Train Model

# COMMAND ----------
# Configure MLflow to use Databricks workspace tracking
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/pneumonia-poc-experiments")

print("MLflow Configuration:")
print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
print(f"  Experiment: /Shared/pneumonia-poc-experiments")

# COMMAND ----------
# Train model with MLflow tracking
print("Starting model training...")
print("This will take approximately 5-10 minutes on CPU cluster...")

with mlflow.start_run(run_name="simple_cnn_poc_v1"):
    # Enable autologging for TensorFlow
    mlflow.tensorflow.autolog()

    # Log hyperparameters
    mlflow.log_param("sample_size", SAMPLE_SIZE_PER_CLASS * 2)  # Total images (50 per class * 2)
    mlflow.log_param("image_size", IMAGE_SIZE)
    mlflow.log_param("model_architecture", "SimpleCNN")
    mlflow.log_param("optimizer", "adam")
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", 10)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=16,
        verbose=1
    )

    # Evaluate on test set (Keras calculates all metrics for us!)
    test_results = model.evaluate(X_test, y_test, verbose=0, return_dict=True)

    # Extract metrics from Keras evaluation
    test_loss = test_results['loss']
    test_accuracy = test_results['accuracy']
    test_precision = test_results['precision']
    test_recall = test_results['recall']
    test_auc = test_results['auc']

    # Calculate F1 score (simple formula: 2 * precision * recall / (precision + recall))
    if test_precision + test_recall > 0:
        test_f1 = 2 * test_precision * test_recall / (test_precision + test_recall)
    else:
        test_f1 = 0.0

    # Log metrics to MLflow
    mlflow.log_metric("test_loss", test_loss)
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1_score", test_f1)
    mlflow.log_metric("test_auc", test_auc)

    # Calculate confusion matrix for logging
    y_pred_probs = model.predict(X_test, verbose=0)
    y_pred = (y_pred_probs > 0.5).astype(int).flatten()
    cm = tf.math.confusion_matrix(y_test, y_pred).numpy()
    mlflow.log_param("confusion_matrix", cm.tolist())

    # Register model to MLflow Model Registry (Unity Catalog)
    mlflow.keras.log_model(
        model,
        artifact_path="model",
        registered_model_name="healthcare_catalog_dev.models.pneumonia_poc_classifier"
    )

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE!")
    print("="*80)
    print(f"\nTest Metrics:")
    print(f"  Loss:      {test_loss:.3f}")
    print(f"  Accuracy:  {test_accuracy:.3f} (Target: >0.50)")
    print(f"  Precision: {test_precision:.3f}")
    print(f"  Recall:    {test_recall:.3f}")
    print(f"  F1 Score:  {test_f1:.3f}")
    print(f"  AUC:       {test_auc:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  {cm}")
    print(f"\nModel registered in MLflow as: healthcare_catalog_dev.models.pneumonia_poc_classifier")
    print(f"\n➡️  Next Step: Deploy serving endpoint (see /Shared/deploy-serving-endpoint)")
