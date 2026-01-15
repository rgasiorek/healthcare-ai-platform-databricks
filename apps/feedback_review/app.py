"""
Radiologist Feedback Review App
Editable table interface for reviewing AI predictions and submitting feedback
"""

import streamlit as st
import pandas as pd
from datetime import datetime
import uuid

# Detect if running in Databricks Apps (use Spark) or locally (use SQL connector)
import os

# Check if running in Databricks environment
IS_DATABRICKS = os.path.exists('/databricks/spark') or 'DATABRICKS_RUNTIME_VERSION' in os.environ

if IS_DATABRICKS:
    # Running in Databricks Apps - use Spark directly
    USE_SQL_CONNECTOR = False
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()
else:
    # Running locally - use SQL connector
    try:
        from databricks import sql
        USE_SQL_CONNECTOR = True
    except ImportError:
        st.error("❌ Running locally but databricks-sql-connector not installed. Run: pip install databricks-sql-connector")
        st.stop()

# Configuration
CATALOG = "healthcare_catalog_dev"
PREDICTIONS_TABLE = f"{CATALOG}.gold.pneumonia_predictions"
FEEDBACK_TABLE = f"{CATALOG}.gold.prediction_feedback"

st.set_page_config(
    page_title="Radiologist Feedback Review",
    page_icon="📋",
    layout="wide"
)

st.title("Radiologist Feedback Review")
st.markdown("**Review AI predictions and provide ground truth diagnosis**")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    radiologist_id = st.text_input("Radiologist ID", value="DR_DEMO_001")

    st.markdown("---")
    st.markdown("### Instructions")
    st.markdown("""
    1. Review predictions in the table
    2. Edit the **editable columns**:
       - Ground Truth
       - Confidence
       - Notes
    3. Click "Submit All Feedback" when done

    **Note**: Only highlighted columns are editable
    """)


# Database connection helper
def get_predictions_for_review():
    """Load predictions that don't have feedback yet"""
    query = f"""
        SELECT
            p.prediction_id,
            p.image_id,
            CASE WHEN p.predicted_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as ai_prediction,
            p.prediction_probability,
            CASE WHEN p.true_label = 1 THEN 'PNEUMONIA' ELSE 'NORMAL' END as actual_diagnosis,
            p.model_name,
            p.predicted_at
        FROM {PREDICTIONS_TABLE} p
        LEFT JOIN {FEEDBACK_TABLE} f ON p.prediction_id = f.prediction_id
        WHERE f.feedback_id IS NULL
        ORDER BY p.predicted_at DESC
        LIMIT 50
    """

    if USE_SQL_CONNECTOR:
        # Use SQL connector (for local Streamlit app)
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        cursor = connection.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        cursor.close()
        connection.close()

        df = pd.DataFrame(rows, columns=columns)
    else:
        # Use Spark (for Databricks Apps)
        spark_df = spark.sql(query)
        df = spark_df.toPandas()

    return df


def save_feedback(feedback_df, radiologist_id):
    """Save feedback to the database"""
    records = []

    for _, row in feedback_df.iterrows():
        ai_prediction = row['ai_prediction']
        ground_truth = row['ground_truth']

        # Determine feedback type
        if ai_prediction == 'PNEUMONIA' and ground_truth == 'PNEUMONIA':
            feedback_type = 'true-positive'
        elif ai_prediction == 'PNEUMONIA' and ground_truth == 'NORMAL':
            feedback_type = 'false-positive'
        elif ai_prediction == 'NORMAL' and ground_truth == 'NORMAL':
            feedback_type = 'true-negative'
        elif ai_prediction == 'NORMAL' and ground_truth == 'PNEUMONIA':
            feedback_type = 'false-negative'
        else:
            continue

        record = {
            'feedback_id': f"fb-{uuid.uuid4().hex[:12]}",
            'prediction_id': row['prediction_id'],
            'timestamp': datetime.now().isoformat(),
            'ground_truth': ground_truth,
            'feedback_type': feedback_type,
            'radiologist_id': radiologist_id,
            'confidence': row['confidence'],
            'feedback_source': 'streamlit_app',
            'notes': row.get('notes', '')
        }
        records.append(record)

    if not records:
        return 0

    # Insert into feedback table
    feedback_insert_df = pd.DataFrame(records)

    if USE_SQL_CONNECTOR:
        # Use SQL connector
        connection = sql.connect(
            server_hostname=st.secrets["databricks"]["server_hostname"],
            http_path=st.secrets["databricks"]["http_path"],
            access_token=st.secrets["databricks"]["access_token"]
        )
        cursor = connection.cursor()

        for record in records:
            insert_sql = f"""
                INSERT INTO {FEEDBACK_TABLE} VALUES (
                    '{record['feedback_id']}',
                    '{record['prediction_id']}',
                    '{record['timestamp']}',
                    '{record['ground_truth']}',
                    '{record['feedback_type']}',
                    '{record['radiologist_id']}',
                    '{record['confidence']}',
                    '{record['feedback_source']}',
                    '{record['notes']}'
                )
            """
            cursor.execute(insert_sql)

        cursor.close()
        connection.close()
    else:
        # Use Spark
        from pyspark.sql.types import StructType, StructField, StringType, TimestampType

        schema = StructType([
            StructField("feedback_id", StringType(), True),
            StructField("prediction_id", StringType(), True),
            StructField("timestamp", StringType(), True),
            StructField("ground_truth", StringType(), True),
            StructField("feedback_type", StringType(), True),
            StructField("radiologist_id", StringType(), True),
            StructField("confidence", StringType(), True),
            StructField("feedback_source", StringType(), True),
            StructField("notes", StringType(), True)
        ])

        spark_feedback_df = spark.createDataFrame(records, schema=schema)
        spark_feedback_df.write.mode('append').saveAsTable(FEEDBACK_TABLE)

    return len(records)


# Main app
try:
    st.markdown("### Edit Feedback")
    st.markdown("**Editable columns**: Ground Truth, Confidence, Notes")

    # Create placeholder/loading indicator
    loading_placeholder = st.empty()
    table_placeholder = st.empty()

    # Show skeleton table with grayed-out placeholder rows
    skeleton_df = pd.DataFrame({
        'prediction_id': ['Loading...', 'Loading...', 'Loading...'],
        'ai_prediction': ['', '', ''],
        'prediction_probability': [0.0, 0.0, 0.0],
        'actual_diagnosis': ['', '', ''],
        'predicted_at': ['', '', ''],
        'ground_truth': ['PNEUMONIA', 'PNEUMONIA', 'PNEUMONIA'],
        'confidence': ['confirmed', 'confirmed', 'confirmed'],
        'notes': ['', '', '']
    })

    # Show loading message
    loading_placeholder.info("Loading predictions from database...")

    # Show skeleton table (disabled)
    with table_placeholder.container():
        st.dataframe(
            skeleton_df,
            column_config={
                "prediction_id": "Prediction ID",
                "ai_prediction": "AI Diagnosis",
                "prediction_probability": st.column_config.NumberColumn("Probability", format="%.3f"),
                "actual_diagnosis": "Actual (Known)",
                "predicted_at": "Timestamp",
                "ground_truth": "Ground Truth",
                "confidence": "Confidence",
                "notes": "Notes"
            },
            hide_index=True,
            use_container_width=True
        )

    # Load actual data
    predictions_df = get_predictions_for_review()

    # Clear loading message
    loading_placeholder.empty()

    if predictions_df.empty:
        table_placeholder.empty()
        st.info("No predictions awaiting feedback. All caught up.")
        st.stop()

    # Prepare editable dataframe
    predictions_df['ground_truth'] = predictions_df['actual_diagnosis']
    predictions_df['confidence'] = 'confirmed'
    predictions_df['notes'] = ''

    # Display readonly columns first, then editable ones
    display_df = predictions_df[[
        'prediction_id',
        'ai_prediction',
        'prediction_probability',
        'actual_diagnosis',
        'predicted_at',
        'ground_truth',
        'confidence',
        'notes'
    ]].copy()

    # Format for display
    display_df['prediction_probability'] = display_df['prediction_probability'].round(3)
    display_df['predicted_at'] = pd.to_datetime(display_df['predicted_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

    # Show success message
    st.success(f"Found {len(predictions_df)} predictions awaiting feedback")

    # Replace skeleton with actual editable table
    with table_placeholder.container():
        edited_df = st.data_editor(
            display_df,
            column_config={
                "prediction_id": st.column_config.TextColumn("Prediction ID", width="medium", disabled=True),
                "ai_prediction": st.column_config.TextColumn("AI Diagnosis", width="small", disabled=True),
                "prediction_probability": st.column_config.NumberColumn("Probability", format="%.3f", width="small", disabled=True),
                "actual_diagnosis": st.column_config.TextColumn("Actual (Known)", width="small", disabled=True),
                "predicted_at": st.column_config.TextColumn("Timestamp", width="medium", disabled=True),
                "ground_truth": st.column_config.SelectboxColumn(
                    "Ground Truth",
                    width="medium",
                    options=["PNEUMONIA", "NORMAL"],
                    required=True
                ),
                "confidence": st.column_config.SelectboxColumn(
                    "Confidence",
                    width="medium",
                    options=["confirmed", "uncertain", "needs_review"],
                    required=True
                ),
                "notes": st.column_config.TextColumn("Notes", width="large")
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed"
        )

    # Submit button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])

    submit_status = st.empty()

    with col2:
        if st.button("Submit All Feedback", type="primary", use_container_width=True):
            if not radiologist_id or radiologist_id.strip() == "":
                submit_status.error("Please enter a Radiologist ID in the sidebar")
            else:
                # Show saving message
                submit_status.info("Saving feedback to database...")

                # Save feedback
                count = save_feedback(edited_df, radiologist_id)

                # Clear saving message
                submit_status.empty()

                if count > 0:
                    submit_status.success(f"Successfully submitted {count} feedback entries")
                    st.info("Refresh the page to load new predictions")
                else:
                    submit_status.warning("No feedback to submit")

    # Show summary statistics
    st.markdown("---")
    st.markdown("### Review Summary")

    col1, col2, col3 = st.columns(3)

    with col1:
        matches = (edited_df['ai_prediction'] == edited_df['ground_truth']).sum()
        st.metric("AI Matches Ground Truth", f"{matches}/{len(edited_df)}")

    with col2:
        confirmed_count = (edited_df['confidence'] == 'confirmed').sum()
        st.metric("Confirmed Diagnoses", confirmed_count)

    with col3:
        has_notes = edited_df['notes'].str.len().gt(0).sum()
        st.metric("With Notes", has_notes)

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)
