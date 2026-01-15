# Import Model Performance Dashboard (1 Click)

Instead of manual setup, import the pre-built dashboard directly.

## Quick Import (2 minutes)

### Step 1: Get Your Warehouse ID

```bash
# Option A: From terraform output
cd terraform
terraform output warehouse_id

# Option B: From Databricks UI
# Go to SQL → Warehouses → Click your warehouse → Copy ID from URL
# URL format: /sql/warehouses/YOUR_WAREHOUSE_ID
```

### Step 2: Update Dashboard File

Edit `model_performance_dashboard_import.lvdash.json`:

```json
{
  "displayName": "Model Performance Comparison",
  "warehouse_id": "PASTE_YOUR_WAREHOUSE_ID_HERE",   <-- Replace this
  ...
}
```

Replace `YOUR_WAREHOUSE_ID` with your actual warehouse ID.

### Step 3: Import Dashboard

1. Open Databricks workspace
2. Navigate to **Workspace** → **Create** → **Dashboard**
3. Click **Import dashboard**
4. Upload `model_performance_dashboard_import.lvdash.json`
5. Click **Import**

Done! Dashboard created with 3 visualizations:
- ML Metrics Table (Precision, Recall, Accuracy)
- Confusion Matrix (Bar chart)
- Performance Trends (Line chart over time)

## Alternative: Get Warehouse ID Automatically

Run this to auto-update the file:

```bash
cd dashboards

# Get warehouse ID from terraform
WAREHOUSE_ID=$(cd ../terraform && terraform output -raw warehouse_id)

# Update the JSON file
sed -i.bak "s/YOUR_WAREHOUSE_ID/$WAREHOUSE_ID/" model_performance_dashboard_import.lvdash.json

echo "Updated! Warehouse ID: $WAREHOUSE_ID"
```

Now just import the file in Databricks UI.

## What Gets Created

### Widget 1: ML Metrics Table
- Shows precision, recall, accuracy by model
- Includes TP, FP, TN, FN counts
- Position: Top left (6x3 grid units)

### Widget 2: Confusion Matrix
- Bar chart of feedback type distribution
- Shows true-positive, false-positive, etc.
- Position: Top right (6x3 grid units)

### Widget 3: Performance Trends
- Line chart showing accuracy/precision/recall over time
- Multi-line visualization
- Position: Bottom (12x3 grid units)

## Data Requirements

The dashboard queries these tables:
- `healthcare_catalog_dev.gold.pneumonia_predictions`
- `healthcare_catalog_dev.gold.prediction_feedback`

To populate with data:
1. Run `generate_sample_predictions.py` notebook (creates predictions)
2. Run `end_to_end_demo.py` Step 4 OR use Streamlit app (submit feedback)

## Customizing the Dashboard

After import, you can:
- Resize widgets (drag corners)
- Rearrange layout (drag widget headers)
- Edit queries (click widget → Edit)
- Add filters/parameters
- Change chart types
- Set auto-refresh schedule

## Troubleshooting

**Error: "Warehouse not found"**
- Check warehouse_id is correct in JSON file
- Verify warehouse exists: SQL → Warehouses

**Error: "Table not found"**
- Run terraform apply (creates tables)
- Verify tables exist: `SHOW TABLES IN healthcare_catalog_dev.gold`

**Dashboard shows no data**
- Tables exist but are empty
- Run notebooks to generate predictions and feedback
- See main README for data generation instructions

## File Format

The `.lvdash.json` extension is Databricks' Lakeview dashboard format:
- JSON structure with pages, layouts, widgets
- Contains SQL queries embedded in the file
- Warehouse ID links to compute resources
- Portable across Databricks workspaces
