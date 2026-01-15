# Radiologist Feedback Review App

Interactive Streamlit app for reviewing AI pneumonia predictions and submitting radiologist feedback.

## Features

- ✅ **Editable Table**: Edit ground truth, confidence, and notes directly in the table
- ✅ **Auto-save**: Changes saved when you click Submit
- ✅ **Read-only columns**: Prediction ID, AI diagnosis, probability, etc.
- ✅ **Validation**: Dropdown selectors for ground truth and confidence
- ✅ **Summary stats**: See matches, confirmed diagnoses, and notes count

## Deployment Options

You have **two options** to run this app:

### **Option 1: Databricks Apps** (Recommended - runs in Databricks workspace)

Databricks Apps lets you deploy Streamlit apps directly in your workspace (like Snowflake Streamlit).

**Requirements:**
- Databricks workspace with Apps enabled (available in most regions)
- Databricks CLI configured

**Deploy:**

```bash
# Navigate to app directory
cd apps/feedback_review

# Deploy to Databricks
databricks apps deploy --source-path . --name radiologist-feedback-review

# Get the app URL
databricks apps list
```

The app will be available at:
```
https://<your-workspace>.cloud.databricks.com/apps/radiologist-feedback-review
```

**Benefits:**
- ✅ Runs inside Databricks (no external hosting)
- ✅ Authenticated automatically
- ✅ Direct Spark access to tables
- ✅ No secrets management needed

---

### **Option 2: Local Streamlit** (runs on your laptop)

Run the app locally on your machine and connect to Databricks remotely.

**Requirements:**
- Python 3.9+
- Databricks SQL Warehouse running

**Setup:**

```bash
# 1. Install dependencies
cd apps/feedback_review
pip install -r requirements.txt

# 2. Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 3. Edit .streamlit/secrets.toml with your Databricks credentials:
#    - server_hostname: Your workspace URL
#    - http_path: SQL warehouse path
#    - access_token: Generate from User Settings > Access Tokens

# 4. Run the app
streamlit run app.py
```

The app will open at http://localhost:8501

**Benefits:**
- ✅ No Databricks Apps needed
- ✅ Run on your laptop
- ✅ Easy to customize and test

---

## Usage

1. **Review predictions**: See all predictions awaiting feedback in the table
2. **Edit feedback columns**:
   - **Ground Truth**: Select PNEUMONIA or NORMAL (pre-filled with actual diagnosis)
   - **Confidence**: Select confirmed/uncertain/needs_review
   - **Notes**: Add optional comments
3. **Set Radiologist ID**: Enter your ID in the sidebar
4. **Submit**: Click "Submit All Feedback" button
5. **Refresh**: Reload page to see new predictions

## Screenshots

**Main Interface:**
```
┌─────────────────────────────────────────────────────────────────────┐
│ Prediction ID         │ AI      │ Prob  │ Actual  │ ✏️ Ground Truth │
├─────────────────────────────────────────────────────────────────────┤
│ pred-a1b2c3d4...      │ PNEUMO  │ 0.892 │ PNEUMO  │ [PNEUMONIA ▼]  │
│ pred-e5f6g7h8...      │ NORMAL  │ 0.234 │ NORMAL  │ [NORMAL ▼]     │
└─────────────────────────────────────────────────────────────────────┘
```

**Editable columns** have dropdown selectors and text inputs that work exactly like a spreadsheet.

## Configuration

Edit `app.py` to customize:
- `CATALOG`: Database catalog name
- `PREDICTIONS_TABLE`: Predictions table path
- `FEEDBACK_TABLE`: Feedback table path

## Troubleshooting

**Databricks Apps:**
- Check if Apps is enabled: `databricks apps list`
- View logs: `databricks apps logs radiologist-feedback-review`
- Redeploy: `databricks apps deploy --source-path . --name radiologist-feedback-review --force`

**Local Streamlit:**
- Connection errors: Verify SQL warehouse is running
- Authentication errors: Regenerate access token
- Missing data: Check table names in `app.py`

## Architecture

```
┌──────────────┐
│  Streamlit   │ ← User edits table
│     App      │
└──────┬───────┘
       │
       ├─ Option 1: Direct Spark (Databricks Apps)
       │            ↓
       │       [Unity Catalog Tables]
       │
       └─ Option 2: SQL Connector (Local)
                    ↓
              [SQL Warehouse]
                    ↓
              [Unity Catalog Tables]
```

## Next Steps

- Add image preview (display X-ray next to prediction)
- Add filtering (by date, model, confidence)
- Add batch approval (approve all confirmed predictions)
- Export to CSV for offline review
