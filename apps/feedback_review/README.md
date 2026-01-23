# Radiologist Feedback Review App

Interactive Streamlit app for reviewing AI pneumonia predictions and submitting radiologist feedback.

## Features

- ✅ **Auto-save**: Changes saved **immediately** when you select a radiologist assessment (no submit button)
- ✅ **Editable Table**: Edit "Radiologist's Assessment" directly in the table via dropdown
- ✅ **Image Viewer**: Click "View X-ray" link to see the X-ray image
- ✅ **Validation**: Dropdown selectors (PNEUMONIA/NORMAL)
- ✅ **Session state caching**: Table doesn't reload on every interaction
- ✅ **Summary stats**: See total predictions and feedback coverage

## Deployment Options

### Option 1: Databricks Apps (Recommended - runs in workspace)

Deploy the app directly in your Databricks workspace.

**Requirements:**
- Databricks workspace with Apps enabled
- Databricks CLI installed and configured
- Appropriate permissions to create apps

**Deployment Steps:**

1. **Create app in Databricks UI**:
   - Navigate to **Compute** → **Apps**
   - Click **Create custom app**
   - Name: `radiologist-feedback-review`
   - The UI will show deployment instructions (follow steps 2-4 below)

2. **Sync app files to workspace**:
   ```bash
   cd apps/feedback_review

   # Sync files to your workspace (replace with your username)
   databricks sync --watch . /Workspace/Users/<your-email>/radiologist-feedback-review
   ```

   **Note**: Use `--watch` flag to auto-sync changes during development, or omit it for one-time sync.

3. **Deploy the app**:
   ```bash
   # First deployment (use full path)
   databricks apps deploy radiologist-feedback-review \
     --source-code-path /Workspace/Users/<your-email>/radiologist-feedback-review

   # Subsequent deploys (can omit path if unchanged)
   databricks apps deploy radiologist-feedback-review
   ```

4. **Access the app**:
   - URL shown in the UI after deployment
   - Or check: `databricks apps get radiologist-feedback-review`

**Troubleshooting** (from app logs):
- **Missing package**: Add to `requirements.txt`
- **Permissions issue**: Give service principal access to Unity Catalog tables
- **Missing environment variable**: Add to `env` section of `app.yaml`
- **Wrong command at startup**: Fix `command` section of `app.yaml`

**Benefits:**
- ✅ Runs inside Databricks (no external hosting)
- ✅ Authenticated automatically via workspace
- ✅ Direct Spark access to tables
- ✅ No secrets management needed
- ✅ Auto-sync during development with `--watch`

---

### Option 2: Local Streamlit (runs on your laptop)

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

---

## Usage

1. **Set Radiologist ID**: Enter your ID in the sidebar (e.g., "DR001")
2. **Review predictions**: See all predictions awaiting feedback in the table
3. **View X-ray**: Click "View X-ray" link to see the image in a separate view
4. **Select assessment**: Click dropdown in "Radiologist's Assessment" column and select PNEUMONIA or NORMAL
5. **Auto-save**: Feedback is **immediately saved** to the database when you make a selection (you'll see a green toast notification)
6. **Refresh**: Click "Load New Predictions" button to see new predictions (already-reviewed items won't show up again)

## Screenshots

**Main Interface:**
```
┌─────────────────────────────────────────────────────────────────────────────┐
│ Prediction ID  │ AI Diagnosis │ Prob  │ Image      │ Radiologist's Assessment │
├─────────────────────────────────────────────────────────────────────────────┤
│ 1234-5678...   │ PNEUMONIA    │ 0.892 │ View X-ray │ [Select... ▼]           │
│ abcd-efgh...   │ NORMAL       │ 0.234 │ View X-ray │ [Select... ▼]           │
└─────────────────────────────────────────────────────────────────────────────┘
```

**Auto-save workflow**:
1. Select "PNEUMONIA" or "NORMAL" from dropdown
2. Green toast notification appears: "✅ Saved assessment for 1234-5678..."
3. Feedback immediately written to `gold.prediction_feedback` table
4. No submit button needed!

## Configuration

Edit `app.py` to customize:
- `CATALOG`: Database catalog name (default: `healthcare_catalog_dev`)
- `PREDICTIONS_TABLE`: Inference logs table (auto-created by endpoint)
- `FEEDBACK_TABLE`: Feedback storage table (default: `gold.prediction_feedback`)

**Key implementation details**:
- Uses `databricks-request-id` header for joining predictions to feedback
- Session state caching prevents table reload on every interaction
- Auto-save tracks `previous_assessments` in session state to detect new selections

## Troubleshooting

**Databricks Apps:**
- **App won't start**: Check logs in Apps UI → Click on app → View Logs
- **Import errors**: Verify all dependencies are in `requirements.txt`
- **Table access errors**: Ensure app has permissions to read `gold.pneumonia_predictions` and `gold.prediction_feedback`
- **Image loading issues**: Verify workspace has access to Unity Catalog volumes

**Local Streamlit:**
- **Connection errors**: Verify SQL warehouse is running in Databricks workspace
- **Authentication errors**: Regenerate access token from User Settings > Access Tokens
- **Missing data**: Check table names in `app.py` match your catalog name (default: `healthcare_catalog_dev`)
- **Image loading issues**: Verify WorkspaceClient credentials in `secrets.toml` have access to Unity Catalog volumes

## Architecture

```
Option 1: Databricks Apps (in workspace)
┌──────────────────────────────────────────────────────────────┐
│  Streamlit App (Databricks Apps)                              │
│  ┌──────────────┬─────────────────┬──────────────────────┐   │
│  │ Auto-save    │ Session State   │ Image Viewer         │   │
│  │ Detection    │ Cache           │ (Files API)          │   │
│  └──────────────┴─────────────────┴──────────────────────┘   │
└──────┬───────────────────────────────────────────────────────┘
       │
       │ Direct Spark (in workspace)
       ↓
  [Unity Catalog Tables]
       - gold.pneumonia_predictions
       - gold.prediction_feedback
       - bronze.kaggle_xray_metadata

Option 2: Local Streamlit (on laptop)
┌──────────────────────────────────────────────────────────────┐
│  Streamlit App (Local)                                        │
│  ┌──────────────┬─────────────────┬──────────────────────┐   │
│  │ Auto-save    │ Session State   │ Image Viewer         │   │
│  │ Detection    │ Cache           │ (Files API)          │   │
│  └──────────────┴─────────────────┴──────────────────────┘   │
└──────┬───────────────────────────────────────────────────────┘
       │
       │ SQL Connector (remote)
       ↓
  [SQL Warehouse]
       ↓
  [Unity Catalog Tables]

**Key Features**:
- Auto-save: Detects new selections via session state tracking
- Image viewer: Looks up file_path from bronze table using image_id (security)
- Dual-mode: Auto-detects Databricks vs local environment
- Files API: Loads images using WorkspaceClient (works in both modes)
```

## Implementation Highlights

### Auto-save Logic
```python
# Track previous assessments in session state
if 'previous_assessments' not in st.session_state:
    st.session_state.previous_assessments = {}

# Detect new assessments after table edit
for idx, row in edited_df.iterrows():
    assessment = row['radiologist_assessment']
    pred_id = row['prediction_id']

    # If assessment was just selected (not null and different from before)
    if pd.notna(assessment) and assessment != '' and \
       st.session_state.previous_assessments.get(pred_id) != assessment:
        # Save to database immediately
        save_feedback_to_db(pred_id, assessment, radiologist_id)
        st.toast(f"✅ Saved assessment for {pred_id[:16]}...", icon="✅")
        # Update tracking
        st.session_state.previous_assessments[pred_id] = assessment
```

### Image Viewer Security
```python
# URL only contains image_id, not full file path
st.markdown(f"[{row['prediction_id'][:16]}...](?image={row['image_id']})")

# Backend looks up full path from bronze table
lookup_query = f"""
    SELECT file_path
    FROM {CATALOG}.bronze.kaggle_xray_metadata
    WHERE image_id = '{image_id}'
"""
```

## Next Steps

- ✅ Image viewer implemented (click prediction ID)
- ✅ Auto-save implemented
- ✅ Session state caching implemented
- [ ] Add filtering (by date, model, confidence)
- [ ] Add batch approval (approve all confirmed predictions)
- [ ] Export to CSV for offline review
