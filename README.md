# Healthcare Data Science with Databricks

X-ray pneumonia classification using Databricks, Delta Lake, and Terraform.

## Overview

This project demonstrates end-to-end healthcare data analytics using:
- **Databricks** - Unified analytics platform
- **Delta Lake** - ACID-compliant data lake storage
- **Terraform** - Infrastructure as Code for reproducible deployments
- **Kaggle** - Source of chest X-ray pneumonia dataset

## Architecture

### Medallion Architecture (Bronze → Silver → Gold)

```
Bronze Layer (Raw Data)
  └─ Kaggle X-ray images
  └─ Dataset metadata

Silver Layer (Cleaned Data)
  └─ Image metadata with labels
  └─ Extracted features

Gold Layer (Business Ready)
  └─ ML predictions
  └─ Model performance metrics
  └─ Aggregated analytics
```

## Infrastructure

### Resources Created by Terraform

- **Serverless SQL Warehouse** - Cost-effective query engine
- **Compute Cluster** - For Python/ML workloads (stopped by default)
- **Delta Tables** - ACID-compliant tables in Hive metastore
- **Notebooks** - Data ingestion, table setup, ML training
- **Secrets** - Secure Kaggle API credentials

### File Structure

```
.
├── provider.tf           # Databricks provider configuration
├── main.tf              # Core resources (secrets, hello-world)
├── catalog.tf           # DBFS directory structure
├── cluster.tf           # Traditional compute cluster (Python/ML)
├── warehouse.tf         # Serverless SQL warehouse
├── tables.tf            # Delta table schemas
├── ingestion.tf         # Kaggle data ingestion notebook
└── README.md            # This file
```

## Getting Started

### Prerequisites

- Terraform >= 1.0
- Databricks workspace on AWS
- Databricks CLI configured (~/.databrickscfg)
- Kaggle account with API credentials

### Deployment

1. **Configure Databricks CLI**
   ```bash
   databricks configure --token
   ```

2. **Deploy Infrastructure**
   ```bash
   terraform init
   terraform plan
   terraform apply
   ```

3. **Run Setup Notebooks**
   - Go to Databricks workspace
   - Navigate to `/Shared/setup-delta-tables`
   - Attach to `healthcare-sql-warehouse`
   - Run all cells

4. **Ingest Data** (when ready)
   - Start the cluster: `databricks clusters start <CLUSTER_ID>`
   - Run `/Shared/ingest-kaggle-xray-data` notebook

## Dataset

**Chest X-Ray Images (Pneumonia)**
Source: [Kaggle Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Binary Classification**: Normal vs Pneumonia
- **Sample Size**: ~1000 images (500 per class for demo)
- **Format**: JPEG images

## Delta Tables

| Layer | Table | Purpose |
|-------|-------|---------|
| Bronze | `bronze_kaggle_dataset_info` | Dataset metadata |
| Silver | `silver_xray_metadata` | Image paths and labels |
| Silver | `silver_image_features` | Extracted ML features |
| Gold | `gold_pneumonia_predictions` | Model predictions |
| Gold | `gold_model_performance` | Model metrics |
| Gold | `gold_dataset_summary` | Aggregated statistics (view) |

## Cost Optimization

- **Serverless SQL Warehouse**: Pay per query (~$0.05-0.15 for typical workload)
- **Compute Cluster**: Auto-terminates after 20 minutes idle
- **Spot Instances**: 60-90% cost reduction
- **Estimated Total**: ~$1-2 for complete demo

## Technology Stack

- **Infrastructure**: Terraform
- **Data Platform**: Databricks on AWS
- **Storage Format**: Delta Lake (Parquet + transaction log)
- **Metastore**: Hive Metastore (default)
- **Compute**:
  - Serverless SQL Warehouse (SQL queries)
  - Traditional cluster (Python/ML)

## Future Enhancements

- [ ] Migrate to Unity Catalog when AWS S3 is configured
- [ ] Add ML model training notebook
- [ ] Create BI dashboard with visualizations
- [ ] Implement CI/CD pipeline
- [ ] Add data quality checks
- [ ] Set up monitoring and alerting

## License

MIT

## Author

Radoslaw Gasiorek
Montrose Software
