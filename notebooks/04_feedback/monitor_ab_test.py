# Databricks notebook source
# MAGIC %md
# MAGIC # Champion vs Challenger Performance Monitoring
# MAGIC
# MAGIC This notebook compares performance between Champion and Challenger models to make
# MAGIC data-driven decisions about model promotion.
# MAGIC
# MAGIC **Purpose**:
# MAGIC - Compare Champion (Keras) vs Challenger (PyTorch) accuracy
# MAGIC - Analyze traffic distribution and feedback coverage
# MAGIC - Provide recommendation: promote, keep testing, or retire Challenger
# MAGIC
# MAGIC **Prerequisites**:
# MAGIC - A/B testing endpoint deployed with inference logging
# MAGIC - Feedback collected via feedback_collector
# MAGIC - Predictions table exists with served_model_name
# MAGIC - Feedback table exists with ground_truth labels
# MAGIC
# MAGIC **Educational Value**:
# MAGIC - Shows how to make data-driven ML decisions
# MAGIC - Demonstrates statistical comparison
# MAGIC - Real-world MLOps monitoring

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 1: Configuration and Imports

# COMMAND ----------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# Set display options
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
sns.set_theme(style="whitegrid")

# Configuration
LOOKBACK_DAYS = 7  # Analyze last 7 days
INFERENCE_TABLE = "healthcare_catalog_dev.gold.pneumonia_classifier_predictions"
FEEDBACK_TABLE = "healthcare_catalog_dev.gold.prediction_feedback"
PERFORMANCE_VIEW = "healthcare_catalog_dev.gold.model_performance_live"

print(f"Configuration:")
print(f"  Lookback Period: {LOOKBACK_DAYS} days")
print(f"  Inference Table: {INFERENCE_TABLE}")
print(f"  Feedback Table: {FEEDBACK_TABLE}")
print(f"  Performance View: {PERFORMANCE_VIEW}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 2: Overall Traffic Distribution

# COMMAND ----------
# Query: How many predictions did each model serve?
traffic_query = f"""
SELECT
    served_model_name,
    COUNT(*) as total_predictions,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 1) as traffic_pct,
    MIN(timestamp_ms) as first_prediction,
    MAX(timestamp_ms) as last_prediction
FROM {INFERENCE_TABLE}
WHERE date >= current_date() - INTERVAL {LOOKBACK_DAYS} DAYS
GROUP BY served_model_name
ORDER BY total_predictions DESC
"""

traffic_df = spark.sql(traffic_query).toPandas()

print(f"\n{'='*80}")
print(f"TRAFFIC DISTRIBUTION (Last {LOOKBACK_DAYS} Days)")
print(f"{'='*80}\n")
print(traffic_df.to_string(index=False))

# Visualization
if not traffic_df.empty:
    plt.figure(figsize=(10, 6))
    plt.bar(traffic_df['served_model_name'], traffic_df['total_predictions'], color=['#1f77b4', '#ff7f0e'])
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('Number of Predictions', fontsize=12)
    plt.title(f'Traffic Distribution - Last {LOOKBACK_DAYS} Days', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 3: Feedback Coverage Analysis

# COMMAND ----------
# Query: How much feedback have we collected for each model?
feedback_coverage_query = f"""
SELECT
    p.served_model_name,
    COUNT(DISTINCT p.request_id) as total_predictions,
    COUNT(DISTINCT f.feedback_id) as feedback_count,
    ROUND(COUNT(DISTINCT f.feedback_id) * 100.0 / COUNT(DISTINCT p.request_id), 1) as feedback_coverage_pct
FROM {INFERENCE_TABLE} p
LEFT JOIN {FEEDBACK_TABLE} f ON p.request_id = f.prediction_id
WHERE p.date >= current_date() - INTERVAL {LOOKBACK_DAYS} DAYS
GROUP BY p.served_model_name
ORDER BY p.served_model_name
"""

coverage_df = spark.sql(feedback_coverage_query).toPandas()

print(f"\n{'='*80}")
print(f"FEEDBACK COVERAGE (Last {LOOKBACK_DAYS} Days)")
print(f"{'='*80}\n")
print(coverage_df.to_string(index=False))

print(f"\n💡 Insight:")
if coverage_df.empty:
    print(f"   ⚠️  No predictions found! Deploy A/B endpoint first.")
elif coverage_df['feedback_count'].sum() == 0:
    print(f"   ⚠️  No feedback collected yet! Use feedback_collector to submit ground truth.")
elif coverage_df['feedback_coverage_pct'].min() < 30:
    print(f"   ⚠️  Low feedback coverage (<30%). Need more data for reliable comparison.")
else:
    print(f"   ✅ Good feedback coverage (>30%). Can make reliable comparisons!")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 4: Champion vs Challenger Accuracy Comparison

# COMMAND ----------
# Query: Core metrics for each model
accuracy_query = f"""
SELECT
    served_model_name,

    -- Sample size
    COUNT(*) as total_with_feedback,

    -- Correctness
    SUM(CASE WHEN is_correct = TRUE THEN 1 ELSE 0 END) as correct_predictions,
    SUM(CASE WHEN is_correct = FALSE THEN 1 ELSE 0 END) as incorrect_predictions,

    -- Accuracy
    ROUND(AVG(CASE WHEN is_correct = TRUE THEN 1.0 WHEN is_correct = FALSE THEN 0.0 END) * 100, 2) as accuracy_pct,

    -- Confusion matrix
    SUM(CASE WHEN confusion_label = 'TRUE_POSITIVE' THEN 1 ELSE 0 END) as true_positives,
    SUM(CASE WHEN confusion_label = 'FALSE_POSITIVE' THEN 1 ELSE 0 END) as false_positives,
    SUM(CASE WHEN confusion_label = 'TRUE_NEGATIVE' THEN 1 ELSE 0 END) as true_negatives,
    SUM(CASE WHEN confusion_label = 'FALSE_NEGATIVE' THEN 1 ELSE 0 END) as false_negatives,

    -- Precision and Recall
    ROUND(
        SUM(CASE WHEN confusion_label = 'TRUE_POSITIVE' THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(CASE WHEN confusion_label IN ('TRUE_POSITIVE', 'FALSE_POSITIVE') THEN 1 ELSE 0 END), 0),
        2
    ) as precision_pct,
    ROUND(
        SUM(CASE WHEN confusion_label = 'TRUE_POSITIVE' THEN 1 ELSE 0 END) * 100.0 /
        NULLIF(SUM(CASE WHEN confusion_label IN ('TRUE_POSITIVE', 'FALSE_NEGATIVE') THEN 1 ELSE 0 END), 0),
        2
    ) as recall_pct

FROM {PERFORMANCE_VIEW}
WHERE prediction_date >= current_date() - INTERVAL {LOOKBACK_DAYS} DAYS
  AND is_correct IS NOT NULL  -- Only include predictions with feedback
GROUP BY served_model_name
ORDER BY accuracy_pct DESC
"""

accuracy_df = spark.sql(accuracy_query).toPandas()

print(f"\n{'='*80}")
print(f"CHAMPION VS CHALLENGER ACCURACY (Last {LOOKBACK_DAYS} Days)")
print(f"{'='*80}\n")

if accuracy_df.empty:
    print("⚠️  No predictions with feedback found!")
    print("   1. Ensure A/B endpoint is deployed and receiving traffic")
    print("   2. Submit feedback using feedback_collector")
else:
    print(accuracy_df.to_string(index=False))

    # Identify champion and challenger
    if len(accuracy_df) >= 2:
        best_model = accuracy_df.iloc[0]['served_model_name']
        best_accuracy = accuracy_df.iloc[0]['accuracy_pct']
        second_model = accuracy_df.iloc[1]['served_model_name']
        second_accuracy = accuracy_df.iloc[1]['accuracy_pct']
        accuracy_delta = best_accuracy - second_accuracy

        print(f"\n🏆 Winner: {best_model} ({best_accuracy}%)")
        print(f"📊 Delta: +{accuracy_delta:.2f}% better than {second_model}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 5: Confusion Matrix Visualization

# COMMAND ----------
if not accuracy_df.empty:
    fig, axes = plt.subplots(1, len(accuracy_df), figsize=(6*len(accuracy_df), 5))

    if len(accuracy_df) == 1:
        axes = [axes]

    for idx, row in accuracy_df.iterrows():
        # Create confusion matrix
        cm = [
            [row['true_negatives'], row['false_positives']],
            [row['false_negatives'], row['true_positives']]
        ]

        # Plot
        sns.heatmap(
            cm,
            annot=True,
            fmt='g',
            cmap='Blues',
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'],
            ax=axes[idx],
            cbar_kws={'label': 'Count'}
        )

        axes[idx].set_title(f"{row['served_model_name']}\nAccuracy: {row['accuracy_pct']}%",
                           fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('True Label', fontsize=10)
        axes[idx].set_xlabel('Predicted Label', fontsize=10)

    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 6: Daily Accuracy Trends

# COMMAND ----------
# Query: Daily accuracy trends
daily_trends_query = f"""
SELECT
    DATE(prediction_date) as date,
    served_model_name,
    COUNT(*) as predictions_with_feedback,
    ROUND(AVG(CASE WHEN is_correct = TRUE THEN 1.0 WHEN is_correct = FALSE THEN 0.0 END) * 100, 2) as accuracy_pct
FROM {PERFORMANCE_VIEW}
WHERE prediction_date >= current_date() - INTERVAL {LOOKBACK_DAYS} DAYS
  AND is_correct IS NOT NULL
GROUP BY DATE(prediction_date), served_model_name
ORDER BY date DESC, served_model_name
"""

daily_df = spark.sql(daily_trends_query).toPandas()

if not daily_df.empty:
    print(f"\n{'='*80}")
    print(f"DAILY ACCURACY TRENDS")
    print(f"{'='*80}\n")
    print(daily_df.to_string(index=False))

    # Line chart
    plt.figure(figsize=(12, 6))
    for model in daily_df['served_model_name'].unique():
        model_data = daily_df[daily_df['served_model_name'] == model]
        plt.plot(model_data['date'], model_data['accuracy_pct'], marker='o', label=model, linewidth=2)

    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    plt.title(f'Daily Accuracy Trends - Last {LOOKBACK_DAYS} Days', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    display(plt.gcf())
    plt.close()

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 7: Statistical Significance Test

# COMMAND ----------
# Chi-square test for statistical significance
from scipy import stats

if not accuracy_df.empty and len(accuracy_df) >= 2:
    print(f"\n{'='*80}")
    print(f"STATISTICAL SIGNIFICANCE TEST")
    print(f"{'='*80}\n")

    # Create contingency table
    model1 = accuracy_df.iloc[0]
    model2 = accuracy_df.iloc[1]

    contingency_table = [
        [model1['correct_predictions'], model1['incorrect_predictions']],
        [model2['correct_predictions'], model2['incorrect_predictions']]
    ]

    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)

    print(f"Model 1: {model1['served_model_name']}")
    print(f"  Correct: {model1['correct_predictions']}, Incorrect: {model1['incorrect_predictions']}")
    print(f"  Accuracy: {model1['accuracy_pct']}%")
    print(f"\nModel 2: {model2['served_model_name']}")
    print(f"  Correct: {model2['correct_predictions']}, Incorrect: {model2['incorrect_predictions']}")
    print(f"  Accuracy: {model2['accuracy_pct']}%")
    print(f"\nChi-Square Test:")
    print(f"  χ² = {chi2:.4f}")
    print(f"  p-value = {p_value:.4f}")
    print(f"  Degrees of freedom = {dof}")

    if p_value < 0.05:
        print(f"\n✅ Result: Statistically SIGNIFICANT difference (p < 0.05)")
        print(f"   The performance difference is likely real, not due to chance.")
    else:
        print(f"\n⚠️  Result: NOT statistically significant (p >= 0.05)")
        print(f"   Cannot confidently say one model is better. Need more data or larger difference.")

# COMMAND ----------
# MAGIC %md
# MAGIC ## Step 8: Recommendation Engine

# COMMAND ----------
print(f"\n{'='*80}")
print(f"RECOMMENDATION: CHAMPION/CHALLENGER DECISION")
print(f"{'='*80}\n")

if accuracy_df.empty:
    print("⚠️  INSUFFICIENT DATA")
    print("\nAction Items:")
    print("  1. Deploy A/B testing endpoint")
    print("  2. Collect predictions (both models should receive traffic)")
    print("  3. Submit feedback using feedback_collector")
    print("  4. Re-run this notebook when feedback coverage > 30%")

elif len(accuracy_df) < 2:
    print("⚠️  ONLY ONE MODEL HAS DATA")
    print("\nAction Items:")
    print("  1. Verify A/B endpoint is splitting traffic correctly")
    print("  2. Check if both models are registered and healthy")
    print("  3. Ensure traffic split is configured (50/50 recommended)")

else:
    best = accuracy_df.iloc[0]
    second = accuracy_df.iloc[1]
    delta = best['accuracy_pct'] - second['accuracy_pct']
    min_sample = min(best['total_with_feedback'], second['total_with_feedback'])

    print(f"📊 Data Quality:")
    print(f"   Sample Size: {min_sample} predictions per model (with feedback)")
    print(f"   Feedback Coverage: {coverage_df['feedback_coverage_pct'].mean():.1f}% average")

    print(f"\n🏆 Performance:")
    print(f"   Winner: {best['served_model_name']} ({best['accuracy_pct']}%)")
    print(f"   Runner-up: {second['served_model_name']} ({second['accuracy_pct']}%)")
    print(f"   Delta: {delta:.2f}%")

    # Decision logic
    if min_sample < 50:
        recommendation = "⏸️  KEEP TESTING (Insufficient sample size)"
        action = f"Continue A/B testing until each model has >= 50 predictions with feedback"
    elif delta < 1.0:
        recommendation = "⏸️  KEEP TESTING (Difference too small)"
        action = f"Models are essentially tied ({delta:.2f}% difference). Continue testing."
    elif delta >= 5.0 and (len(accuracy_df) < 2 or p_value < 0.05):
        recommendation = f"✅ PROMOTE {best['served_model_name']} TO CHAMPION"
        action = f"""
Gradual Rollout Plan:
  Week 1: Current (50/50 split) - Continue monitoring
  Week 2: Shift to 70/30 in favor of {best['served_model_name']}
  Week 3: Shift to 90/10 in favor of {best['served_model_name']}
  Week 4: Promote {best['served_model_name']} to 100% (Champion)

Update traffic split in deploy_ab_testing_endpoint notebook:
  CHAMPION_TRAFFIC_PCT = 70  # {best['served_model_name']}
  CHALLENGER_TRAFFIC_PCT = 30  # {second['served_model_name']}
"""
    elif delta >= 1.0:
        recommendation = f"⚠️  CAUTIOUSLY FAVOR {best['served_model_name']}"
        action = f"""
Moderate difference ({delta:.2f}%). Proceed carefully:
  - Increase {best['served_model_name']} to 70% traffic
  - Monitor for 3-5 days
  - If trend continues, promote further

Update traffic split:
  CHAMPION_TRAFFIC_PCT = 70  # {best['served_model_name']}
  CHALLENGER_TRAFFIC_PCT = 30  # {second['served_model_name']}
"""

    print(f"\n💡 RECOMMENDATION: {recommendation}")
    print(f"\n📋 ACTION PLAN:")
    print(action)

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary: Monitoring Dashboard Complete
# MAGIC
# MAGIC ### What This Notebook Provides:
# MAGIC 1. ✅ Traffic distribution analysis
# MAGIC 2. ✅ Feedback coverage metrics
# MAGIC 3. ✅ Champion vs Challenger accuracy comparison
# MAGIC 4. ✅ Confusion matrices
# MAGIC 5. ✅ Daily accuracy trends
# MAGIC 6. ✅ Statistical significance testing
# MAGIC 7. ✅ Automated recommendation engine
# MAGIC
# MAGIC ### How to Use:
# MAGIC 1. Deploy A/B testing endpoint (Issue #11)
# MAGIC 2. Collect predictions (both models receive traffic)
# MAGIC 3. Submit feedback via feedback_collector (Issue #13)
# MAGIC 4. Run this notebook daily/weekly
# MAGIC 5. Follow recommendation to adjust traffic or promote winner
# MAGIC
# MAGIC ### Decision Criteria:
# MAGIC - **Sample Size**: >= 50 predictions per model with feedback
# MAGIC - **Significant Win**: Delta >= 5% + statistically significant → Promote
# MAGIC - **Moderate Win**: Delta 1-5% → Favor winner, continue testing
# MAGIC - **Tie**: Delta < 1% → Keep testing
# MAGIC - **Insufficient Data**: Sample < 50 → Keep testing
# MAGIC
# MAGIC ### Educational Value:
# MAGIC - Shows data-driven decision making in MLOps
# MAGIC - Demonstrates A/B testing methodology
# MAGIC - Explains statistical significance
# MAGIC - Models gradual rollout strategy
# MAGIC
# MAGIC **Perfect for teaching pupils how to make production ML decisions!**
