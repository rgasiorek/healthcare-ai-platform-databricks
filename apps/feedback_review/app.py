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

# Check if this is an image viewer request
query_params = st.query_params
if "image" in query_params:
    # Image viewer mode - receives only image_id (not full path)
    image_id = query_params["image"]

    st.title("X-Ray Image Viewer")
    st.markdown(f"**Image ID:** {image_id}")

    try:
        # Show loading spinner while fetching image
        with st.spinner("Loading X-ray image..."):
            # Look up full path from bronze table using image_id
            lookup_query = f"""
                SELECT file_path, filename
                FROM {CATALOG}.bronze.kaggle_xray_metadata
                WHERE image_id = '{image_id}'
                LIMIT 1
            """

            if USE_SQL_CONNECTOR:
                connection = sql.connect(
                    server_hostname=st.secrets["databricks"]["server_hostname"],
                    http_path=st.secrets["databricks"]["http_path"],
                    access_token=st.secrets["databricks"]["access_token"]
                )
                cursor = connection.cursor()
                cursor.execute(lookup_query)
                result = cursor.fetchone()
                cursor.close()
                connection.close()

                if not result:
                    st.error(f"Image not found: {image_id}")
                    st.stop()

                image_path, filename = result
            else:
                result = spark.sql(lookup_query).collect()
                if not result:
                    st.error(f"Image not found: {image_id}")
                    st.stop()

                image_path = result[0].file_path
                filename = result[0].filename

            # Load the image using Files API (works both locally and in Databricks Apps)
            from databricks.sdk import WorkspaceClient
            from PIL import Image
            from io import BytesIO

            # Remove dbfs: prefix for Files API
            clean_path = image_path.replace("dbfs:", "")

            # Use Files API to download (with credentials for local mode)
            if IS_DATABRICKS:
                # In Databricks Apps: use default auth
                w = WorkspaceClient()
            else:
                # Running locally: use credentials from secrets
                # Extract workspace host from SQL warehouse hostname (e.g., "dbc-xxx.cloud.databricks.com")
                sql_hostname = st.secrets["databricks"]["server_hostname"]
                workspace_host = sql_hostname.split('/')[0] if '/' in sql_hostname else sql_hostname

                w = WorkspaceClient(
                    host=f"https://{workspace_host}",
                    token=st.secrets["databricks"]["access_token"]
                )

            file_content = w.files.download(clean_path).contents.read()

            # Display image
            img = Image.open(BytesIO(file_content))
            st.image(img, caption=filename, use_container_width=True)

    except Exception as e:
        st.error(f"Failed to load image: {e}")
        import traceback
        st.code(traceback.format_exc())

    st.stop()

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
    2. Click image link to view X-ray
    3. Select **Radiologist's Assessment** (PNEUMONIA or NORMAL)

    **Note**: Assessments are saved immediately to database
    """)

    st.markdown("---")
    if st.button("🔄 Load New Predictions", use_container_width=True):
        st.session_state.reload_data = True
        if 'display_df' in st.session_state:
            del st.session_state.display_df
        if 'previous_assessments' in st.session_state:
            del st.session_state.previous_assessments
        st.rerun()


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
            p.predicted_at,
            -- Get image path from bronze table using image_id
            bronze.file_path AS image_path
        FROM {PREDICTIONS_TABLE} p
        LEFT JOIN {FEEDBACK_TABLE} f ON p.prediction_id = f.prediction_id
        LEFT JOIN {CATALOG}.bronze.kaggle_xray_metadata bronze
            ON p.image_id = bronze.image_id
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
        radiologist_assessment = row['radiologist_assessment']

        # Skip rows where assessment wasn't selected
        if pd.isna(radiologist_assessment) or radiologist_assessment is None or radiologist_assessment == '':
            continue

        # Determine feedback type
        if ai_prediction == 'PNEUMONIA' and radiologist_assessment == 'PNEUMONIA':
            feedback_type = 'true-positive'
        elif ai_prediction == 'PNEUMONIA' and radiologist_assessment == 'NORMAL':
            feedback_type = 'false-positive'
        elif ai_prediction == 'NORMAL' and radiologist_assessment == 'NORMAL':
            feedback_type = 'true-negative'
        elif ai_prediction == 'NORMAL' and radiologist_assessment == 'PNEUMONIA':
            feedback_type = 'false-negative'
        else:
            continue

        record = {
            'feedback_id': f"fb-{uuid.uuid4().hex[:12]}",
            'prediction_id': row['prediction_id'],
            'timestamp': datetime.now().isoformat(),
            'ground_truth': radiologist_assessment,
            'feedback_type': feedback_type,
            'radiologist_id': radiologist_id,
            'confidence': 'confirmed',  # Always confirmed for dashboard compatibility
            'feedback_source': 'streamlit_app',
            'notes': ''  # No notes captured in UI
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


# Initialize session state for predictions data
if 'reload_data' not in st.session_state:
    st.session_state.reload_data = False

# Check if we need to load data
if 'predictions_df' not in st.session_state or st.session_state.reload_data:
    # Show loading overlay with grayed-out skeleton table
    loading_container = st.empty()

    with loading_container.container():
        # Create overlay effect with grayed-out skeleton table
        st.markdown("""
            <style>
            .loading-overlay {
                position: relative;
                opacity: 0.3;
            }
            </style>
        """, unsafe_allow_html=True)

        st.markdown('<div class="loading-overlay">', unsafe_allow_html=True)

        # Skeleton table
        skeleton_df = pd.DataFrame({
            'prediction_id': ['...', '...', '...', '...', '...'],
            'ai_prediction': ['', '', '', '', ''],
            'prediction_probability': [0.0, 0.0, 0.0, 0.0, 0.0],
            'actual_diagnosis': ['', '', '', '', ''],
            'predicted_at': ['', '', '', '', ''],
            'image_link': ['', '', '', '', ''],
            'radiologist_assessment': [None, None, None, None, None]
        })

        st.dataframe(
            skeleton_df,
            column_config={
                "prediction_id": "Prediction ID",
                "ai_prediction": "AI Diagnosis",
                "prediction_probability": st.column_config.NumberColumn("Probability", format="%.3f"),
                "actual_diagnosis": "Actual (Known)",
                "predicted_at": "Timestamp",
                "image_link": "Image",
                "radiologist_assessment": "Radiologist's Assessment"
            },
            hide_index=True,
            use_container_width=True
        )

        st.markdown('</div>', unsafe_allow_html=True)

        # Loading message overlay
        st.markdown("""
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%);
                        background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                        font-size: 18px; font-weight: bold;">
                Loading predictions...
            </div>
        """, unsafe_allow_html=True)

    # Load actual data
    st.session_state.predictions_df = get_predictions_for_review()
    st.session_state.reload_data = False

    # Clear loading overlay
    loading_container.empty()
    st.rerun()

predictions_df = st.session_state.predictions_df

# Main app
try:
    st.markdown("### Review Predictions")
    st.markdown("**Select assessment for each prediction - saved automatically | Click image to view X-ray**")

    if predictions_df.empty:
        st.info("No predictions awaiting feedback. All caught up.")
        st.stop()

    # Prepare display dataframe (only once, store in session state)
    if 'display_df' not in st.session_state:
        df_copy = predictions_df.copy()
        df_copy['radiologist_assessment'] = None  # Start empty - radiologist must select

        # Create image viewer links with query parameters (only pass image_id, not full path)
        df_copy['image_link'] = df_copy['image_id'].apply(
            lambda x: f"?image={x}" if pd.notna(x) and x else None
        )

        # Display readonly columns first, then editable ones
        display_df = df_copy[[
            'prediction_id',
            'ai_prediction',
            'prediction_probability',
            'actual_diagnosis',
            'predicted_at',
            'image_link',
            'radiologist_assessment'
        ]].copy()

        # Format for display
        display_df['prediction_probability'] = display_df['prediction_probability'].round(3)
        display_df['predicted_at'] = pd.to_datetime(display_df['predicted_at']).dt.strftime('%Y-%m-%d %H:%M:%S')

        st.session_state.display_df = display_df

    # Show success message
    st.success(f"Found {len(predictions_df)} predictions awaiting feedback")

    # Editable table
    edited_df = st.data_editor(
            st.session_state.display_df,
            column_config={
                "prediction_id": st.column_config.TextColumn("Prediction ID", width="medium", disabled=True),
                "ai_prediction": st.column_config.TextColumn("AI Diagnosis", width="small", disabled=True),
                "prediction_probability": st.column_config.NumberColumn("Probability", format="%.3f", width="small", disabled=True),
                "actual_diagnosis": st.column_config.TextColumn("Actual (Known)", width="small", disabled=True),
                "predicted_at": st.column_config.TextColumn("Timestamp", width="medium", disabled=True),
                "image_link": st.column_config.LinkColumn("Image", width="medium", disabled=True, display_text="View X-ray"),
                "radiologist_assessment": st.column_config.SelectboxColumn(
                    "Radiologist's Assessment",
                    width="medium",
                    options=["PNEUMONIA", "NORMAL"],
                    required=False
                )
            },
            hide_index=True,
            use_container_width=True,
            num_rows="fixed",
            key="predictions_editor"
        )

    # Auto-save: detect changes and save immediately
    if 'previous_assessments' not in st.session_state:
        st.session_state.previous_assessments = {}

    # Check for new assessments
    new_assessments = []
    for idx, row in edited_df.iterrows():
        assessment = row['radiologist_assessment']
        pred_id = row['prediction_id']

        # If assessment was just selected (not null and different from before)
        if pd.notna(assessment) and assessment != '' and st.session_state.previous_assessments.get(pred_id) != assessment:
            new_assessments.append((idx, row))
            st.session_state.previous_assessments[pred_id] = assessment

    # Save new assessments immediately
    if new_assessments and radiologist_id and radiologist_id.strip() != "":
        for idx, row in new_assessments:
            # Save single feedback entry
            ai_prediction = row['ai_prediction']
            radiologist_assessment = row['radiologist_assessment']

            # Determine feedback type
            if ai_prediction == 'PNEUMONIA' and radiologist_assessment == 'PNEUMONIA':
                feedback_type = 'true-positive'
            elif ai_prediction == 'PNEUMONIA' and radiologist_assessment == 'NORMAL':
                feedback_type = 'false-positive'
            elif ai_prediction == 'NORMAL' and radiologist_assessment == 'NORMAL':
                feedback_type = 'true-negative'
            elif ai_prediction == 'NORMAL' and radiologist_assessment == 'PNEUMONIA':
                feedback_type = 'false-negative'
            else:
                continue

            record = {
                'feedback_id': f"fb-{uuid.uuid4().hex[:12]}",
                'prediction_id': row['prediction_id'],
                'timestamp': datetime.now().isoformat(),
                'ground_truth': radiologist_assessment,
                'feedback_type': feedback_type,
                'radiologist_id': radiologist_id,
                'confidence': 'confirmed',
                'feedback_source': 'streamlit_app',
                'notes': ''
            }

            # Save to database
            if USE_SQL_CONNECTOR:
                connection = sql.connect(
                    server_hostname=st.secrets["databricks"]["server_hostname"],
                    http_path=st.secrets["databricks"]["http_path"],
                    access_token=st.secrets["databricks"]["access_token"]
                )
                cursor = connection.cursor()
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
                from pyspark.sql.types import StructType, StructField, StringType
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
                spark_feedback_df = spark.createDataFrame([record], schema=schema)
                spark_feedback_df.write.mode('append').saveAsTable(FEEDBACK_TABLE)

            # Show success toast
            st.toast(f"✅ Saved assessment for {row['prediction_id'][:16]}...", icon="✅")

    # Show summary statistics
    st.markdown("---")
    st.markdown("### Review Summary")

    col1, col2 = st.columns(2)

    with col1:
        # Count assessments selected
        assessed = edited_df['radiologist_assessment'].notna().sum()
        st.metric("Assessments Selected", f"{assessed}/{len(edited_df)}")

    with col2:
        # Count matches between AI and radiologist (only for assessed rows)
        assessed_df = edited_df[edited_df['radiologist_assessment'].notna()]
        if len(assessed_df) > 0:
            matches = (assessed_df['ai_prediction'] == assessed_df['radiologist_assessment']).sum()
            st.metric("AI Matches Assessment", f"{matches}/{len(assessed_df)}")
        else:
            st.metric("AI Matches Assessment", "0/0")

except Exception as e:
    st.error(f"Error: {str(e)}")
    st.exception(e)
