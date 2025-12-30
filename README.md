# Healthcare AI Platform - Databricks on AWS

Production-ready X-ray pneumonia classification platform using Databricks Unity Catalog, Delta Lake, and Terraform Infrastructure as Code.

## Overview

This project demonstrates end-to-end healthcare data engineering and ML using:
- **Unity Catalog** - Unified governance for data and AI assets
- **Delta Lake** - ACID-compliant lakehouse storage on S3
- **Terraform** - Infrastructure as Code for reproducible deployments
- **AWS S3** - Scalable object storage for data lake
- **AWS IAM** - Secure cross-account access control
- **Kaggle** - Chest X-ray pneumonia dataset source

## Architecture

### Unity Catalog Three-Level Namespace

```
healthcare_catalog_dev
├── bronze                           # Raw data layer
│   ├── kaggle_xray_metadata        # Raw X-ray metadata from Kaggle
│   └── xray_images (volume)        # Raw JPEG image files
├── silver                          # Cleaned data layer
│   ├── xray_metadata               # Validated X-ray metadata with labels
│   └── image_features              # Extracted CNN features
└── gold                            # Business-ready layer
    ├── pneumonia_predictions       # ML model predictions
    └── model_performance           # Model evaluation metrics
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
| **Notebooks** | 2 notebooks | Hello World + Kaggle ingestion |

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

## Project Tracking

All work tracked via GitHub Issues:

- ✅ [Issue #1](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/1): Setup AWS integration for Unity Catalog
- ✅ [Issue #2](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/2): Implement production-ready Delta tables
- ⏳ [Issue #3](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/3): Ingest Kaggle X-ray data into Bronze
- 📋 [Issue #4](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/4): Implement Bronze → Silver transformation
- 📋 [Issue #5](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/5): Create BI dashboard for analytics
- 📋 [Issue #6](https://github.com/rgasiorek/healthcare-ai-platform-databricks/issues/6): Implement ML model for pneumonia classification

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

- [x] Unity Catalog with AWS S3 integration
- [x] Environment-based naming (dev/pilot/prod)
- [x] Production-ready Delta tables with Terraform
- [x] Kaggle data ingestion pipeline
- [ ] Bronze → Silver transformation with data quality checks
- [ ] BI dashboard with Databricks SQL
- [ ] ML model training (CNN for pneumonia classification)
- [ ] MLflow model registry integration
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Monitoring and alerting

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
