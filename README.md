# Healthcare AI Platform - Databricks on AWS

End-to-end demonstration of Databricks capabilities on AWS, showcasing a complete MLOps platform for healthcare AI with pneumonia classification from chest X-rays.

## Project Goals

### Primary Goal: Comprehensive Databricks Platform Exercise

This project exercises the **complete spectrum of Databricks capabilities** in a single cohesive platform:

**Data Engineering & Lakehouse:**
- **Data Ingestion** - Automated Kaggle dataset download via Databricks Jobs
- **Unity Catalog** - Three-tier namespace (catalog → schema → table/volume) with governance
- **Delta Lake** - ACID transactions, time travel, change data feed on S3
- **Medallion Architecture** - Bronze (raw) → Silver (validated) → Gold (business-ready)
- **External Volumes** - Unity Catalog managed volumes for unstructured data (JPEG images)

**ML & MLOps:**
- **Model Training** - Distributed training with TensorFlow (Keras) and PyTorch
- **MLflow Experiments** - Experiment tracking, model versioning, hyperparameter logging
- **Model Registry** - Unity Catalog-backed model registry (UC models, not workspace models)
- **Model Serving** - REST API endpoints with A/B testing (Champion vs Challenger, 50/50 traffic split)
- **Inference Logging** - Auto-capture prediction payloads for monitoring and debugging

**Analytics & Applications:**
- **SQL Warehouses** - Serverless compute for BI and analytics
- **Lakeview Dashboards** - 7-widget performance dashboard with ML metrics, confusion matrices, radiologist agreement analysis
- **Databricks Apps** - Streamlit feedback review app hosted in Databricks workspace

**Serving Diverse Stakeholders:**
- **ML Analysts** - Experiment tracking, model comparison, performance metrics
- **ML Engineers** - Model training notebooks, serving endpoint deployment, inference monitoring
- **Data Engineers** - ETL pipelines, data quality, medallion architecture
- **MLOps Engineers** - A/B testing infrastructure, feedback loops, model governance
- **BI/Data Analysts** - SQL-based dashboards, business metrics, radiologist performance tracking
- **Senior Leadership** - High-level KPIs, model ROI, accuracy trends over time
- **External Vendors** - REST API integration, feedback submission workflows

### Secondary Goal: LLM-Driven Development Without UI Clicking

**"Make it without clicking around, only with LLM agent with agile processes in place"**

This project was built entirely through conversation with **Claude Code** (AI assistant with CLI access), demonstrating:

**Terraform as Formal Language for LLM:**
- Terraform serves dual purpose: Infrastructure as Code + AI context
- LLM translates natural language requirements → Terraform HCL
- AI reads existing `.tf` files → generates correct table schemas, column names, resource dependencies
- No trial-and-error: AI understands Unity Catalog structure from Terraform state

**Virtual Junior Developer in CICD:**
- LLM authenticates to GitHub CLI, commits code, creates issues, tracks tickets
- Works within real agile processes: tickets → commits → code reviews → testing
- Tracks all work via [GitHub Issues](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues)
- Plugged into CICD: Terraform plans, git workflows, deployment automation

**Zero-Click Automation:**
- 100% automation: conversation → code → deployed infrastructure
- No AWS console clicking, no Databricks UI clicking for infrastructure
- Human role: strategic decisions, architecture approval, domain knowledge
- AI role: execution, documentation, ticket tracking, code generation

**Result**: Staff engineer experiments with production MLOps patterns in days, not months

## Architecture Overview

This platform demonstrates a complete production-ready MLOps system with:
- **End-to-End ML Pipeline** - Data ingestion → Training → Serving → Monitoring → Feedback
- **A/B Testing Infrastructure** - Champion/Challenger pattern with traffic splitting
- **Feedback Loop** - Real-world accuracy tracking with radiologist ground truth
- **Comprehensive Dashboards** - 7 widgets tracking model performance, confusion matrices, radiologist agreement
- **Production-Grade Governance** - Unity Catalog access control, Delta Lake ACID transactions

## Architecture

### Unity Catalog Three-Level Namespace

```
healthcare_catalog_dev
├── bronze                           # Raw data layer
│   ├── kaggle_xray_metadata        # Raw X-ray metadata from Kaggle
│   └── xray_images (volume)        # Raw JPEG image files (~1000 X-rays)
├── models                          # ML models layer
│   ├── pneumonia_poc_classifier    # TensorFlow/Keras CNN model
│   └── pneumonia_poc_classifier_pytorch  # PyTorch CNN model
└── gold                            # Business-ready layer
    ├── prediction_feedback         # Ground truth labels from radiologists
    ├── pneumonia_classifier_predictions  # Inference table (auto-logged)
    ├── pneumonia_classifier_payload      # Inference table (auto-logged)
    └── model_performance_live (view)     # Real-time accuracy metrics
```

### Medallion Architecture (Bronze → Silver → Gold)

```
┌─────────────────────────────────────────────────────────────────┐
│ BRONZE LAYER (Raw Data)                                         │
│ S3: s3://healthcare-data-lake-dev/bronze/                       │
├─────────────────────────────────────────────────────────────────┤
│ • Kaggle X-ray images (Unity Catalog volume)                    │
│ • kaggle_xray_metadata (8 columns)                              │
│ • Immutable, append-only                                        │
│ • Change Data Feed enabled                                      │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ SILVER LAYER (Cleaned & Validated Data)                         │
│ S3: s3://healthcare-data-lake-dev/silver/                       │
├─────────────────────────────────────────────────────────────────┤
│ • xray_metadata: Validated metadata with quality checks         │
│ • image_features: CNN-extracted features (ARRAY<DOUBLE>)        │
│ • Auto-optimize enabled (compaction + optimize writes)          │
│ • Data quality validation applied                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ GOLD LAYER (Business-Ready Analytics)                           │
│ S3: s3://healthcare-data-lake-dev/gold/                         │
├─────────────────────────────────────────────────────────────────┤
│ • pneumonia_predictions: ML predictions with probabilities      │
│ • model_performance: Accuracy, precision, recall, F1, AUC-ROC   │
│ • Optimized for BI dashboards and reporting                     │
└─────────────────────────────────────────────────────────────────┘
```

### AWS Integration

```
AWS Account (905418100642)
├── S3 Buckets
│   ├── healthcare-databricks-unity-catalog-dev  # Unity Catalog metastore
│   └── healthcare-data-lake-dev                 # Data lake (bronze/silver/gold)
│
├── IAM Roles (with self-assuming trust policies)
│   ├── databricks-unity-catalog-metastore-dev   # Metastore access
│   └── databricks-healthcare-data-access-dev    # Data lake access
│
└── Cross-Account Trust
    └── Databricks Unity Catalog Service (414351767826)
        └── AssumeRole with ExternalId (Databricks Account ID)
```

## Infrastructure

### Resources Created by Terraform

| Resource Type | Resource Name | Purpose |
|--------------|---------------|---------|
| **AWS S3** | `healthcare-databricks-unity-catalog-dev` | Unity Catalog metastore storage |
| **AWS S3** | `healthcare-data-lake-dev` | Healthcare data lake (medallion layers) |
| **AWS IAM** | `databricks-unity-catalog-metastore-dev` | IAM role for metastore access |
| **AWS IAM** | `databricks-healthcare-data-access-dev` | IAM role for data lake access |
| **Unity Catalog** | `healthcare-metastore-dev-eu-central-1` | Unity Catalog metastore |
| **Unity Catalog** | `healthcare_catalog_dev` | Healthcare catalog |
| **Schemas** | `bronze`, `silver`, `gold` | Medallion architecture layers |
| **Storage Credentials** | 2 credentials | Map IAM roles to Databricks |
| **External Locations** | 3 locations | Map S3 paths to Unity Catalog |
| **Delta Tables** | 5 tables | Production-ready table schemas |
| **Volume** | `xray_images` | External volume for JPEG files |
| **Compute Cluster** | `healthcare-data-cluster-dev` | Python/ML workloads (2x i3.xlarge) |
| **SQL Warehouse** | Serverless warehouse | SQL queries and BI dashboards |
| **Notebooks** | 6 notebooks | Ingestion, training, model wrapping, demos |
| **Jobs** | 1 Databricks Job | Automated Kaggle data ingestion |
| **Model Endpoints** | 1 A/B testing endpoint | Champion vs Challenger with 50/50 traffic split |
| **Streamlit App** | Feedback review app | Interactive radiologist feedback collection |

### File Structure

```
.
├── variables.tf          # Environment configuration (dev/pilot/prod)
├── provider.tf           # Databricks + AWS provider configuration
├── aws_provider.tf       # AWS provider (separate for modularity)
├── aws_s3.tf            # S3 buckets for Unity Catalog + data lake
├── aws_iam.tf           # IAM roles and policies with trust relationships
├── catalog.tf           # Unity Catalog: metastore, catalog, schemas, volumes
├── tables.tf            # Delta table definitions (5 tables)
├── cluster.tf           # Databricks compute cluster
├── warehouse.tf         # Databricks SQL warehouse
├── ingestion.tf         # Kaggle data ingestion notebook
├── main.tf              # Kaggle secrets
├── IAM_SETUP.md         # AWS IAM setup guide
└── README.md            # This file
```

## Delta Tables

| Layer | Table Name | Columns | Purpose |
|-------|-----------|---------|---------|
| **Bronze** | `kaggle_xray_metadata` | 8 | Raw X-ray metadata from Kaggle |
| **Silver** | `xray_metadata` | 10 | Cleaned metadata with labels and quality scores |
| **Silver** | `image_features` | 6 | CNN-extracted features (ARRAY<DOUBLE>) |
| **Gold** | `pneumonia_predictions` | 11 | ML predictions with probabilities and validation |
| **Gold** | `model_performance` | 13 | Model metrics (accuracy, precision, recall, F1, AUC-ROC) |

All tables:
- ✅ S3-backed external storage
- ✅ Change Data Feed enabled
- ✅ Managed by Terraform (not SQL notebooks)
- ✅ Three-level Unity Catalog namespace
- ✅ Proper Delta Lake properties

## Getting Started

### Prerequisites

- **Terraform** >= 1.0
- **Databricks workspace** on AWS (E2 architecture)
- **AWS CLI** configured with SSO or credentials
- **Databricks CLI** configured (`~/.databrickscfg`)
- **Kaggle account** with API credentials
- **AWS permissions**: IAM role creation, S3 bucket management

### Step-by-Step Deployment

#### 1. Configure Databricks CLI

```bash
databricks configure --token
# Enter workspace URL: https://dbc-68a1cdfa-43b8.cloud.databricks.com
# Enter personal access token: (generate from User Settings → Developer → Access Tokens)
```

#### 2. Configure AWS CLI

```bash
# For AWS SSO
aws sso login --profile DevAdmin-905418100642

# For static credentials
aws configure --profile DevAdmin-905418100642
```

#### 3. Update Variables

Edit `variables.tf`:
```hcl
variable "environment" {
  default = "dev"  # or "pilot", "prod"
}

variable "databricks_account_id" {
  default = "YOUR_DATABRICKS_ACCOUNT_ID"
}
```

Edit `provider.tf`:
```hcl
provider "aws" {
  profile = "YOUR_AWS_PROFILE"
  region  = "eu-central-1"
}
```

#### 4. Deploy Infrastructure

```bash
terraform init
terraform plan
terraform apply
```

**What gets created:**
- 2 S3 buckets (metastore + data lake)
- 2 IAM roles with self-assuming policies
- Unity Catalog metastore + catalog + schemas
- 5 Delta tables (bronze/silver/gold)
- 1 Unity Catalog volume
- 1 compute cluster + 1 SQL warehouse
- 6 notebooks
- 1 ingestion job
- 1 model serving endpoint (A/B testing)

**Deployment time:** ~5-7 minutes

#### 5. Configure Kaggle Credentials

The Kaggle secret scope was already created by Terraform. Verify in Databricks:

```bash
databricks secrets list-secrets --scope kaggle
```

If you need to update credentials:
```bash
databricks secrets put-secret --scope kaggle --key username --string-value YOUR_KAGGLE_USERNAME
databricks secrets put-secret --scope kaggle --key token --string-value YOUR_KAGGLE_API_TOKEN
```

#### 6. Run Data Ingestion (Issue #3)

1. Go to Databricks Workspace → **Shared** → **ingest-kaggle-xray-data**
2. Attach to cluster: `healthcare-data-cluster-dev`
3. Click **Run All**
4. Wait ~10-15 minutes (downloads 1000 X-ray images from Kaggle)

**What happens:**
- Downloads Chest X-Ray Pneumonia dataset from Kaggle
- Stores 1000 JPEG images in Unity Catalog volume
- Writes metadata to `bronze.kaggle_xray_metadata` table
- Tracks batch with unique `ingestion_batch_id`

## Dataset

**Chest X-Ray Images (Pneumonia)**
Source: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Classification**: Binary (Normal vs Pneumonia)
- **Sample Size**: ~1000 images (500 per class for demo)
- **Format**: JPEG grayscale X-ray images
- **Dataset Splits**: train, test, val

## Environment-Based Naming Convention

All resources use environment postfix for multi-environment support:

| Environment | Suffix | Use Case |
|------------|--------|----------|
| `dev` | `-dev` | Development and testing |
| `pilot` | `-pilot` | Staging/pre-production |
| `prod` | `-prod` | Production workloads |

**Example:**
- S3: `healthcare-data-lake-dev` → `healthcare-data-lake-pilot` → `healthcare-data-lake-prod`
- IAM: `databricks-healthcare-data-access-dev` → `...-pilot` → `...-prod`
- Catalog: `healthcare_catalog_dev` → `healthcare_catalog_pilot` → `healthcare_catalog_prod`

To deploy to different environment: Update `variables.tf` → `environment = "pilot"` → `terraform apply`

## MLOps Pipeline

### Complete ML Workflow

```
1. TRAIN
   └─► Train models (Keras Champion + PyTorch Challenger)
       Register in MLflow Model Registry with Unity Catalog

2. DEPLOY
   └─► Create A/B testing endpoint (Databricks Model Serving)
       Champion: 50% traffic | Challenger: 50% traffic
       Enable inference logging (auto_capture)

3. PREDICT
   └─► REST API: Make predictions
       Capture request_id from response headers
       Return prediction + request_id to user

4. COLLECT FEEDBACK
   └─► Radiologist reviews X-ray (hours/days later)
       submit_feedback(request_id, "true-positive", ...)
       Stored in prediction_feedback table

5. ANALYZE
   └─► JOIN inference_table + feedback_table
       Calculate per-model accuracy
       Statistical significance testing (Chi-square)

6. DECIDE & PROMOTE
   └─► Challenger is better? Promote to Champion
       Update traffic: Challenger 90%, New_Challenger 10%
       Continuous improvement cycle
```

### Models & Endpoints

| Model | Framework | Use Case | Status |
|-------|-----------|----------|--------|
| `pneumonia_poc_classifier` | TensorFlow/Keras | Champion model | ✅ Deployed |
| `pneumonia_poc_classifier_pytorch` | PyTorch | Challenger model | ✅ Deployed |

**Serving Endpoints**:
- `pneumonia-poc-classifier` - Single model endpoint
- `pneumonia-classifier-ab-test` - A/B testing endpoint (50/50 split)

### Feedback Infrastructure

| Component | Purpose |
|-----------|---------|
| `prediction_feedback` table | Ground truth labels from radiologists |
| `pneumonia_predictions` table | AI predictions with confidence scores |
| **Streamlit App** (`apps/feedback_review/app.py`) | Interactive table-based feedback review interface |
| **SQL Dashboard** (`dashboards/model_comparison_dashboard.sql`) | Real-time model comparison with ML metrics |

## Notebooks

All notebooks are deployed via Terraform to `/Shared/` in Databricks workspace.

| Notebook | Purpose | Location |
|----------|---------|----------|
| `ingest_kaggle_xray_data` | Download and ingest X-ray dataset from Kaggle | `/notebooks/01_ingestion/` |
| `train_poc_model` | Train TensorFlow/Keras CNN model (Champion) | `/notebooks/03_ml/` |
| `train_poc_model_pytorch` | Train PyTorch CNN model (Challenger) | `/notebooks/03_ml/` |
| `wrap_and_register_path_models` | Wrap models for path-based inference (Files API) | `/notebooks/03_ml/` |
| `demo_model_usage` | Demo SDK vs REST API for model inference | `/notebooks/03_ml/` |
| `end_to_end_demo` | Complete end-to-end A/B testing workflow demo | `/notebooks/05_demo/` |

**Note**: Model serving endpoints are deployed via Terraform (`terraform/databricks/endpoints.tf`), not notebooks. See `DEPLOYMENT.md` for details.

## Feedback & Monitoring

### Streamlit Feedback Review App

Interactive table-based interface for radiologists to review AI predictions:

```bash
cd apps/feedback_review
pip install -r requirements.txt
streamlit run app.py
```

**Features**:
- **Auto-save**: Changes saved immediately on selection (no submit button)
- **Editable table**: Dropdown selectors for radiologist assessments
- **Image viewer**: Click on prediction ID to view X-ray image
- **Real-time validation**: Validates input before saving
- **Direct database writes**: Saves to `gold.prediction_feedback` table
- **Automatic categorization**: Calculates feedback type (TP/FP/TN/FN)

**Deployment Options**:
- **Option 1**: Run locally with `streamlit run app.py` (requires `.streamlit/secrets.toml`)
- **Option 2**: Deploy to Databricks Apps with `databricks apps deploy` (recommended)

See `apps/feedback_review/README.md` for full setup instructions.

### Model Comparison Dashboard

SQL-based dashboard for real-time model performance monitoring:

**Location**: `/dashboards/model_comparison_dashboard.sql`

**Metrics Displayed**:
- Precision, Recall, F1 Score, Accuracy (per model)
- Confusion matrix (TP, FP, TN, FN)
- Performance trends over time
- Confidence analysis (correct vs incorrect predictions)
- Error analysis (false positives vs false negatives)
- Radiologist feedback summary

**How to Create**: See `/dashboards/README.md` for step-by-step instructions

## Project Tracking

All work tracked via [GitHub Issues](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues).

Development followed agile methodology with ticket-driven workflow managed entirely through Claude Code CLI integration.

## Exploring the Platform

### Databricks UI

1. **Unity Catalog**: **Catalog** → `healthcare_catalog_dev` → Browse schemas/tables
2. **Tables**: Click any table → See Schema, Sample Data, Details, History, Properties
3. **Volume**: **bronze** → **xray_images** → Browse uploaded JPEG files
4. **External Locations**: **Catalog** → **External Data** → See S3 mappings
5. **Storage Credentials**: **Catalog** → **Storage Credentials** → See IAM role ARNs
6. **SQL Editor**: Run queries against tables (see "Quick Queries" below)
7. **Notebooks**: **Workspace** → **Shared** → See ingestion pipeline

### AWS Console

1. **S3 Buckets**: See `healthcare-databricks-unity-catalog-dev` and `healthcare-data-lake-dev`
2. **IAM Roles**: See `databricks-unity-catalog-metastore-dev` and `databricks-healthcare-data-access-dev`
3. **Trust Policies**: Check self-assuming configurations in IAM role trust relationships
4. **S3 Data**: Browse medallion layers (bronze/silver/gold) after ingestion

### Quick Queries

Run in Databricks SQL Editor:

```sql
-- List all tables
SHOW TABLES IN healthcare_catalog_dev.bronze;
SHOW TABLES IN healthcare_catalog_dev.silver;
SHOW TABLES IN healthcare_catalog_dev.gold;

-- Describe table schema
DESCRIBE EXTENDED healthcare_catalog_dev.bronze.kaggle_xray_metadata;

-- Check table properties
SHOW TBLPROPERTIES healthcare_catalog_dev.bronze.kaggle_xray_metadata;

-- Query data (after ingestion)
SELECT category, COUNT(*) as count
FROM healthcare_catalog_dev.bronze.kaggle_xray_metadata
GROUP BY category;

-- Check Delta Lake history
DESCRIBE HISTORY healthcare_catalog_dev.bronze.kaggle_xray_metadata;
```

## Cost Optimization

- **S3 Storage**: ~$0.023/GB/month (first 50 TB)
- **Serverless SQL Warehouse**: Pay per query (~$0.05-0.15 for typical demo)
- **Compute Cluster**: Auto-terminates after 20 minutes idle
- **Spot Instances**: Available for cost reduction
- **Unity Catalog**: No additional cost (included in Databricks)

**Estimated Total for Demo**: ~$2-5 (depends on cluster runtime)

## Technology Stack

| Layer | Technology |
|-------|-----------|
| **Infrastructure** | Terraform (IaC) |
| **Cloud Platform** | AWS (S3, IAM) |
| **Data Platform** | Databricks on AWS |
| **Governance** | Unity Catalog |
| **Storage Format** | Delta Lake (Parquet + transaction log) |
| **Compute** | Databricks Clusters + Serverless SQL |
| **Data Architecture** | Medallion (Bronze/Silver/Gold) |
| **Security** | AWS IAM roles with self-assuming policies |
| **Version Control** | Git + GitHub |

## Security Features

✅ **No Databricks-managed storage** - All data in your own S3 buckets
✅ **IAM cross-account access** - Secure AssumeRole with ExternalId
✅ **Self-assuming IAM roles** - Unity Catalog requirement met
✅ **S3 encryption** - SSE-S3 enabled on all buckets
✅ **S3 versioning** - Enabled for data protection
✅ **Private buckets** - Public access blocked
✅ **Unity Catalog governance** - Centralized access control
✅ **Secrets management** - Kaggle credentials in Databricks Secrets

## Roadmap

**Completed**:
- [x] Unity Catalog with AWS S3 integration
- [x] Environment-based naming (dev/pilot/prod)
- [x] Production-ready Delta tables with Terraform
- [x] Kaggle data ingestion pipeline (1000 X-rays)
- [x] ML model training (TensorFlow + PyTorch CNNs)
- [x] MLflow model registry integration (Unity Catalog)
- [x] Model serving endpoints (single + A/B testing)
- [x] Champion/Challenger A/B testing infrastructure
- [x] Feedback loop system with ground truth tracking
- [x] Monitoring dashboard for model comparison

**Future Enhancements**:
- [ ] Bronze → Silver transformation with data quality checks
- [ ] Feature engineering pipeline
- [ ] Transfer learning with EfficientNet/ResNet
- [ ] Hyperparameter tuning with Databricks AutoML
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Automated model retraining on feedback
- [ ] Real-time alerting for model drift
- [ ] BI dashboard with Databricks SQL
- [ ] Multi-class classification (normal/bacterial/viral)

## Troubleshooting

### IAM Permission Issues

If you see `cannot create sql table` or IAM errors:
1. Ensure AWS profile has `iam:CreateRole`, `iam:PutRolePolicy` permissions
2. Verify `databricks_account_id` in `variables.tf` is correct
3. Check IAM role trust policies include self-assuming statements

### Table Creation Errors

If you see `DELTA_CREATE_TABLE_WITH_DIFFERENT_PROPERTY`:
1. Databricks auto-adds properties like `delta.writePartitionColumnsToParquet`
2. Ensure all properties in `tables.tf` match what Databricks expects
3. If needed, destroy and recreate: `terraform destroy -target=databricks_sql_table.RESOURCE_NAME`

### Unity Catalog Not Showing

If catalog doesn't appear:
1. Check metastore assignment: Workspace must be assigned to metastore
2. Verify in Databricks: **Settings** → **Data** → **Metastores**
3. Check workspace ID in `catalog.tf` matches your workspace

## License

MIT

## Author

**Radoslaw Gasiorek**

---

🤖 Infrastructure managed with [Claude Code](https://claude.com/claude-code)
