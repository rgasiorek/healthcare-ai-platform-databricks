# Databricks notebook source
# MAGIC %md
# MAGIC # Pneumonia Classification POC - Train Model (PyTorch)
# MAGIC
# MAGIC **PyTorch version - for comparison with TensorFlow approach**
# MAGIC
# MAGIC This notebook trains the SAME simple CNN classifier but using PyTorch:
# MAGIC 1. Load 100 images directly from filesystem (50 NORMAL, 50 PNEUMONIA)
# MAGIC 2. Preprocess and split into train/val/test sets
# MAGIC 3. Train simple CNN model (~5-10 min on CPU)
# MAGIC 4. Register model in MLflow Model Registry
# MAGIC
# MAGIC **Why PyTorch?**
# MAGIC - More explicit control (great for learning ML fundamentals)
# MAGIC - Industry standard for research and modern ML
# MAGIC - Preferred by AI researchers (e.g., Andrej Karpathy)
# MAGIC
# MAGIC **Key Differences from TensorFlow**:
# MAGIC - Explicit training loop (no `model.fit()`)
# MAGIC - Manual forward/backward pass
# MAGIC - nn.Module class definition
# MAGIC - DataLoader for batching
# MAGIC
# MAGIC **Goal**: Same as TensorFlow version - prove workflow works. Accept any accuracy.
# MAGIC
# MAGIC **Next Steps**: Deploy registered model as serving endpoint (see `/Shared/deploy-serving-endpoint`)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Install Dependencies

# COMMAND ----------
# Install required libraries (PyTorch, Pillow, and MLflow)
%pip install torch==2.1.0 torchvision==0.16.0 Pillow mlflow --quiet

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

# PyTorch (explicit control over training!)
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# MLflow
import mlflow
import mlflow.pytorch

# Configuration
VOLUME_PATH = "/Volumes/healthcare_catalog_dev/bronze/xray_images"  # Unity Catalog volume (no /dbfs prefix!)
SAMPLE_SIZE_PER_CLASS = 50  # 50 NORMAL + 50 PNEUMONIA = 100 total
IMAGE_SIZE = 64    # 64x64 for fast training
TRAIN_SPLIT = 0.70  # 70% train
VAL_SPLIT = 0.15    # 15% validation
TEST_SPLIT = 0.15   # 15% test

# Device configuration (CPU or GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Configuration:")
print(f"  Volume: {VOLUME_PATH}")
print(f"  Sample Size: {SAMPLE_SIZE_PER_CLASS * 2} images (50 per class)")
print(f"  Image Size: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"  Splits: {TRAIN_SPLIT:.0%} train, {VAL_SPLIT:.0%} val, {TEST_SPLIT:.0%} test")
print(f"  Device: {device}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Load Images Directly from Filesystem
# MAGIC
# MAGIC Same as TensorFlow version - no Spark needed!

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
# MAGIC ## Step 7: Create PyTorch Dataset and DataLoader
# MAGIC
# MAGIC **PyTorch Difference #1**: Need to create custom Dataset class and DataLoader

# COMMAND ----------
class XRayDataset(Dataset):
    """Custom PyTorch Dataset for X-ray images"""

    def __init__(self, images, labels):
        # Convert to PyTorch tensors
        # PyTorch expects (batch, channels, height, width) format
        # NumPy has (batch, height, width, channels), so we permute
        self.images = torch.FloatTensor(images).permute(0, 3, 1, 2)
        self.labels = torch.FloatTensor(labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

# Create PyTorch datasets
train_dataset = XRayDataset(X_train, y_train)
val_dataset = XRayDataset(X_val, y_val)
test_dataset = XRayDataset(X_test, y_test)

# Create DataLoaders (handles batching automatically)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

print(f"DataLoaders created:")
print(f"  Train batches: {len(train_loader)}")
print(f"  Val batches: {len(val_loader)}")
print(f"  Test batches: {len(test_loader)}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 8: Define CNN Model Architecture
# MAGIC
# MAGIC **PyTorch Difference #2**: Explicit class definition with `__init__` and `forward` methods

# COMMAND ----------
class SimpleCNN(nn.Module):
    """Simple CNN for binary classification - PyTorch style"""

    def __init__(self, image_size=64):
        super(SimpleCNN, self).__init__()

        # First convolutional block
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Second convolutional block
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Calculate flattened size after convolutions
        # Input: 64x64 -> Conv(3x3): 62x62 -> Pool(2x2): 31x31
        #             -> Conv(3x3): 29x29 -> Pool(2x2): 14x14
        # Output: 64 channels * 14 * 14 = 12544
        self.fc1 = nn.Linear(64 * 14 * 14, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        """Forward pass - explicit definition of how data flows through network"""
        # First conv block
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Second conv block
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Flatten
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch (more robust than view)

        # Dense layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc2(x))

        return x

# Initialize model and move to device (CPU or GPU)
model = SimpleCNN(image_size=IMAGE_SIZE).to(device)

# Display model architecture
print("Model Architecture:")
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 9: Define Loss Function and Optimizer
# MAGIC
# MAGIC **PyTorch Difference #3**: Explicit loss function and optimizer definition

# COMMAND ----------
# Binary Cross-Entropy loss for binary classification
criterion = nn.BCELoss()

# Adam optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"Training configuration:")
print(f"  Loss function: Binary Cross-Entropy")
print(f"  Optimizer: Adam (lr=0.001)")
print(f"  Device: {device}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 10: Configure MLflow

# COMMAND ----------
# Configure MLflow to use Databricks workspace tracking
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Shared/pneumonia-poc-experiments-pytorch")

print("MLflow Configuration:")
print(f"  Tracking URI: {mlflow.get_tracking_uri()}")
print(f"  Experiment: /Shared/pneumonia-poc-experiments-pytorch")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 11: Training Functions
# MAGIC
# MAGIC **PyTorch Difference #4**: Manual training loop (no `model.fit()`)

# COMMAND ----------
def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch - PyTorch style"""
    model.train()  # Set model to training mode

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        # Move data to device
        images = images.to(device)
        labels = labels.to(device).view(-1, 1)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Track statistics
        running_loss += loss.item() * images.size(0)
        predicted = (outputs > 0.5).float()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate for one epoch - PyTorch style"""
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient computation for validation
        for images, labels in dataloader:
            # Move data to device
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Track statistics
            running_loss += loss.item() * images.size(0)
            predicted = (outputs > 0.5).float()
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

def calculate_metrics(model, dataloader, device):
    """Calculate detailed metrics (precision, recall, F1, AUC)"""
    model.eval()

    all_labels = []
    all_predictions = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device).view(-1, 1)

            outputs = model(images)
            predicted = (outputs > 0.5).float()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            all_probs.extend(outputs.cpu().numpy())

    all_labels = np.array(all_labels).flatten()
    all_predictions = np.array(all_predictions).flatten()
    all_probs = np.array(all_probs).flatten()

    # Calculate metrics manually
    tp = np.sum((all_labels == 1) & (all_predictions == 1))
    fp = np.sum((all_labels == 0) & (all_predictions == 1))
    tn = np.sum((all_labels == 0) & (all_predictions == 0))
    fn = np.sum((all_labels == 1) & (all_predictions == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # Simple AUC calculation (for educational purposes)
    # In production, use sklearn.metrics.roc_auc_score
    sorted_indices = np.argsort(all_probs)[::-1]
    sorted_labels = all_labels[sorted_indices]

    # Count inversions (pairs where positive is ranked higher than negative)
    n_pos = np.sum(all_labels == 1)
    n_neg = np.sum(all_labels == 0)

    if n_pos == 0 or n_neg == 0:
        auc = 0.5
    else:
        inversions = 0
        for i, label in enumerate(sorted_labels):
            if label == 1:
                inversions += np.sum(sorted_labels[i+1:] == 0)
        auc = inversions / (n_pos * n_neg)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': [[int(tn), int(fp)], [int(fn), int(tp)]]
    }

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 12: Train Model with MLflow Tracking
# MAGIC
# MAGIC **PyTorch Difference #5**: Manual epoch loop with explicit logging

# COMMAND ----------
print("Starting model training...")
print("This will take approximately 5-10 minutes on CPU cluster...")
print(f"\n{'='*80}")

EPOCHS = 10

with mlflow.start_run(run_name="simple_cnn_pytorch_poc_v1"):
    # Enable autologging for PyTorch
    mlflow.pytorch.autolog()

    # Log hyperparameters
    mlflow.log_param("sample_size", SAMPLE_SIZE_PER_CLASS * 2)
    mlflow.log_param("image_size", IMAGE_SIZE)
    mlflow.log_param("model_architecture", "SimpleCNN_PyTorch")
    mlflow.log_param("optimizer", "Adam")
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("epochs", EPOCHS)
    mlflow.log_param("framework", "PyTorch")

    # Training loop
    for epoch in range(EPOCHS):
        # Train one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # Validate one epoch
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

        # Log metrics to MLflow
        mlflow.log_metric("train_loss", train_loss, step=epoch)
        mlflow.log_metric("train_accuracy", train_acc, step=epoch)
        mlflow.log_metric("val_loss", val_loss, step=epoch)
        mlflow.log_metric("val_accuracy", val_acc, step=epoch)

        print(f"Epoch {epoch+1}/{EPOCHS} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} - "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # Evaluate on test set
    print(f"\n{'='*80}")
    print("Calculating final test metrics...")

    test_metrics = calculate_metrics(model, test_loader, device)

    # Log test metrics
    mlflow.log_metric("test_accuracy", test_metrics['accuracy'])
    mlflow.log_metric("test_precision", test_metrics['precision'])
    mlflow.log_metric("test_recall", test_metrics['recall'])
    mlflow.log_metric("test_f1_score", test_metrics['f1'])
    mlflow.log_metric("test_auc", test_metrics['auc'])
    mlflow.log_param("confusion_matrix", test_metrics['confusion_matrix'])

    # Create model signature (REQUIRED for Unity Catalog)
    # Signature defines input/output schema for serving endpoint
    from mlflow.models import infer_signature

    # Get sample input and output for signature inference
    sample_input = X_test[:1]  # Single sample (1, 64, 64, 3) NumPy array
    sample_input_tensor = torch.FloatTensor(sample_input).permute(0, 3, 1, 2).to(device)

    model.eval()
    with torch.no_grad():
        sample_output = model(sample_input_tensor).cpu().numpy()

    # Infer signature from sample data
    signature = infer_signature(sample_input, sample_output)

    # Register model to MLflow Model Registry (Unity Catalog)
    # Note: Unity Catalog REQUIRES signature for all models
    mlflow.pytorch.log_model(
        model,
        artifact_path="model",
        registered_model_name="healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch",
        signature=signature  # REQUIRED for Unity Catalog
    )

    print(f"{'='*80}")
    print("✅ TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"\nTest Metrics:")
    print(f"  Accuracy:  {test_metrics['accuracy']:.3f} (Target: >0.50)")
    print(f"  Precision: {test_metrics['precision']:.3f}")
    print(f"  Recall:    {test_metrics['recall']:.3f}")
    print(f"  F1 Score:  {test_metrics['f1']:.3f}")
    print(f"  AUC:       {test_metrics['auc']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  [[TN={test_metrics['confusion_matrix'][0][0]}, FP={test_metrics['confusion_matrix'][0][1]}],")
    print(f"   [FN={test_metrics['confusion_matrix'][1][0]}, TP={test_metrics['confusion_matrix'][1][1]}]]")
    print(f"\nModel registered in MLflow as: healthcare_catalog_dev.models.pneumonia_poc_classifier_pytorch")
    print(f"\n➡️  Next Step: Deploy serving endpoint (see /Shared/deploy-serving-endpoint)")
    print(f"\n💡 Educational Note:")
    print(f"  Compare this PyTorch code with the TensorFlow version:")
    print(f"  - PyTorch: Explicit training loop, manual forward/backward pass")
    print(f"  - TensorFlow: High-level model.fit() API")
    print(f"  Both achieve the same goal - PyTorch gives you more control!")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary: PyTorch vs TensorFlow
# MAGIC
# MAGIC ### Key Differences for Your Pupils:
# MAGIC
# MAGIC | Aspect | TensorFlow/Keras | PyTorch |
# MAGIC |--------|------------------|---------|
# MAGIC | **API Style** | High-level, declarative | Lower-level, imperative |
# MAGIC | **Model Definition** | `Sequential()` | `nn.Module` class |
# MAGIC | **Training** | `model.fit()` | Manual epoch loop |
# MAGIC | **Forward Pass** | Implicit | Explicit `forward()` method |
# MAGIC | **Gradient** | Automatic | Manual `backward()` + `zero_grad()` |
# MAGIC | **Data Format** | (batch, H, W, C) | (batch, C, H, W) |
# MAGIC | **Best For** | Quick prototyping | Research, custom models |
# MAGIC
# MAGIC ### When to Use Each:
# MAGIC
# MAGIC **TensorFlow/Keras**:
# MAGIC - Beginners learning ML concepts
# MAGIC - Production deployment (mature ecosystem)
# MAGIC - Quick prototyping
# MAGIC
# MAGIC **PyTorch**:
# MAGIC - Research and experimentation
# MAGIC - Custom architectures
# MAGIC - Learning ML fundamentals (more explicit)
# MAGIC - Modern AI development (industry trend)
# MAGIC
# MAGIC ### MLflow Integration:
# MAGIC
# MAGIC Both frameworks work seamlessly with MLflow:
# MAGIC - `mlflow.keras.log_model()` for TensorFlow
# MAGIC - `mlflow.pytorch.log_model()` for PyTorch
# MAGIC - Same serving endpoint API
# MAGIC - Same Model Registry
# MAGIC
# MAGIC ### Discussion Questions for Pupils:
# MAGIC
# MAGIC 1. Which approach gives you more control over the training process?
# MAGIC 2. Why might PyTorch's explicit style be better for learning?
# MAGIC 3. When would you prefer TensorFlow's `model.fit()` simplicity?
# MAGIC 4. How does the data format differ (channels first vs last)?
# MAGIC 5. What's the trade-off between ease of use and flexibility?
