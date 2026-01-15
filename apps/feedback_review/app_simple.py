"""
Simple Streamlit test app for Databricks Apps
"""
import streamlit as st

st.title("🏥 Test App")
st.write("If you see this, Streamlit is working!")

# Test Spark connection
try:
    from pyspark.sql import SparkSession
    spark = SparkSession.builder.getOrCreate()

    st.success("✅ Spark session available")

    # Try a simple query
    df = spark.sql("SELECT 1 as test")
    st.write("Test query result:", df.toPandas())

except Exception as e:
    st.error(f"❌ Spark error: {e}")
