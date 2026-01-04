# Healthcare AI Platform - Databricks on AWS

Production-ready X-ray pneumonia classification platform using Databricks Unity Catalog, Delta Lake, and Terraform Infrastructure as Code.

**Built with AI-Assisted Development**: This project showcases not just a healthcare AI platform, but how **Claude Code** (AI assistant with direct GitHub, AWS, and Databricks access) can accelerate development for staff engineers.

## Overview

This project demonstrates a complete production-ready MLOps platform for healthcare AI, featuring:
- **End-to-End ML Pipeline** - Train → Deploy → Monitor → Improve
- **A/B Testing** - Champion/Challenger pattern with traffic splitting
- **Feedback Loop** - Real-world accuracy tracking with ground truth
- **Unity Catalog** - Unified governance for data and AI assets
- **Delta Lake** - ACID-compliant lakehouse storage on S3
- **Model Serving** - REST API endpoints for real-time predictions
- **MLflow** - Experiment tracking and model registry
- **Terraform** - Infrastructure as Code for reproducible deployments

## AI-Assisted Development with Claude Code

### The Real Innovation: Zero-Click Automation

**Project Goal**: Eliminate clicking in AWS/Databricks consoles - achieve 100% automation for development workflow.

This project demonstrates how **Claude Code with direct system access** (GitHub CLI, AWS CLI, Databricks CLI, Terraform) enables:

### 1. Full SDLC Automation
- **Plan**: GitHub issues created from conversations
- **Build**: Terraform + notebooks written with proper context
- **Deploy**: Direct execution to AWS and Databricks
- **Test**: AI runs tests OR guides user with closed-ended test scenarios matching requirements
- **Document**: README, code comments, presentation generated
- **Maintain**: Issue tracking, commits, closures automated

**Result**: Zero manual clicking in cloud consoles. Testing either automated or guided with clear pass/fail criteria. Everything versioned, reproducible, documented.

### 2. MLOps Experimentation Enabled
- Rapidly test different patterns (A/B testing, feedback loops)
- Try multiple frameworks (TensorFlow vs PyTorch) with minimal friction
- Iterate on architecture decisions with immediate deployment
- Learn Databricks features through AI explanations

**Result**: Staff engineer experiments with production MLOps patterns in days, not months.

### 3. Terraform as AI Context
**Dual purpose**:
- Traditional: Infrastructure as Code (reproducibility)
- Novel: Perfect structured context for AI (reads schemas, writes correct code)

**Result**: No trial-and-error. AI reads `catalog.tf` → writes queries with correct column names first try.

### The Strategic Shift

```
TRADITIONAL:                    AI-ASSISTED:
├─ Click AWS console        →  Terraform apply
├─ Upload files manually    →  Automated deployment
├─ Write docs later         →  Auto-generated docs
├─ Context lost             →  Git + Terraform preserve all
└─ Weeks to MVP             →  Days to MVP
```

**Human role**: Strategic decisions, architecture, approval, domain knowledge
**AI role**: SDLC automation, zero-click execution, complete documentation

**The goal was zero clicking. Achieved: 100% automation from conversation to deployed infrastructure.**

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
| **Notebooks** | 9 notebooks | Ingestion, training, deployment, monitoring, demo |
| **Jobs** | 2 Databricks Jobs | Model deployment automation |

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
- 2 notebooks

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
| `model_performance_live` view | Real-time accuracy calculation |
| `feedback_collector.py` | BentoML-style feedback API |
| `monitor_ab_test.py` | Champion vs Challenger comparison dashboard |

## Project Tracking

All work tracked via GitHub Issues:

**Infrastructure & Data**:
- ✅ [Issue #1](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/1): Setup AWS integration for Unity Catalog
- ✅ [Issue #2](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/2): Implement production-ready Delta tables
- ✅ [Issue #3](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/3): Ingest Kaggle X-ray data into Bronze

**ML Pipeline**:
- ✅ [Issue #6](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/6): Implement ML model POC (TensorFlow)
- ✅ [Issue #7](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/7): Fix serverless endpoint cold start timeout
- ✅ [Issue #8](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/8): Fix REST API payload format for Keras
- ✅ [Issue #9](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/9): Add PyTorch model for framework comparison
- ✅ [Issue #10](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/10): Fix Unity Catalog model signature requirement

**A/B Testing & Feedback Loop**:
- ✅ [Issue #11](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/11): Implement Champion/Challenger A/B testing
- ✅ [Issue #12](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/12): Create feedback infrastructure tables
- ✅ [Issue #13](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/13): Build BentoML-style feedback collector API
- ✅ [Issue #14](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/14): Create monitoring dashboard for model comparison
- ✅ [Issue #15](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/15): Update demo notebook with prediction tracking

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
