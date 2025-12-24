#!/bin/bash
# Migrate flat Terraform state to modular structure

# AWS Module resources
terraform state mv 'data.aws_caller_identity.current' 'module.aws.data.aws_caller_identity.current'
terraform state mv 'data.aws_region.current' 'module.aws.data.aws_region.current'
terraform state mv 'data.aws_iam_policy_document.databricks_assume_role' 'module.aws.data.aws_iam_policy_document.databricks_assume_role'
terraform state mv 'data.aws_iam_policy_document.healthcare_data_s3' 'module.aws.data.aws_iam_policy_document.healthcare_data_s3'
terraform state mv 'data.aws_iam_policy_document.unity_catalog_metastore_s3' 'module.aws.data.aws_iam_policy_document.unity_catalog_metastore_s3'
terraform state mv 'aws_iam_role.healthcare_data_access' 'module.aws.aws_iam_role.healthcare_data_access'
terraform state mv 'aws_iam_role.unity_catalog_metastore' 'module.aws.aws_iam_role.unity_catalog_metastore'
terraform state mv 'aws_iam_role_policy.healthcare_data_s3' 'module.aws.aws_iam_role_policy.healthcare_data_s3'
terraform state mv 'aws_iam_role_policy.unity_catalog_metastore_s3' 'module.aws.aws_iam_role_policy.unity_catalog_metastore_s3'
terraform state mv 'aws_s3_bucket.healthcare_data' 'module.aws.aws_s3_bucket.healthcare_data'
terraform state mv 'aws_s3_bucket.unity_catalog_metastore' 'module.aws.aws_s3_bucket.unity_catalog_metastore'
terraform state mv 'aws_s3_bucket_lifecycle_configuration.unity_catalog_metastore' 'module.aws.aws_s3_bucket_lifecycle_configuration.unity_catalog_metastore'
terraform state mv 'aws_s3_bucket_public_access_block.healthcare_data' 'module.aws.aws_s3_bucket_public_access_block.healthcare_data'
terraform state mv 'aws_s3_bucket_public_access_block.unity_catalog_metastore' 'module.aws.aws_s3_bucket_public_access_block.unity_catalog_metastore'
terraform state mv 'aws_s3_bucket_server_side_encryption_configuration.healthcare_data' 'module.aws.aws_s3_bucket_server_side_encryption_configuration.healthcare_data'
terraform state mv 'aws_s3_bucket_server_side_encryption_configuration.unity_catalog_metastore' 'module.aws.aws_s3_bucket_server_side_encryption_configuration.unity_catalog_metastore'
terraform state mv 'aws_s3_bucket_versioning.healthcare_data' 'module.aws.aws_s3_bucket_versioning.healthcare_data'
terraform state mv 'aws_s3_bucket_versioning.unity_catalog_metastore' 'module.aws.aws_s3_bucket_versioning.unity_catalog_metastore'

# Databricks Module resources
terraform state mv 'databricks_catalog.healthcare' 'module.databricks.databricks_catalog.healthcare'
terraform state mv 'databricks_cluster.healthcare_compute' 'module.databricks.databricks_cluster.healthcare_compute'
terraform state mv 'databricks_external_location.bronze' 'module.databricks.databricks_external_location.bronze'
terraform state mv 'databricks_external_location.gold' 'module.databricks.databricks_external_location.gold'
terraform state mv 'databricks_external_location.silver' 'module.databricks.databricks_external_location.silver'
terraform state mv 'databricks_metastore.healthcare' 'module.databricks.databricks_metastore.healthcare'
terraform state mv 'databricks_metastore_assignment.workspace' 'module.databricks.databricks_metastore_assignment.workspace'
terraform state mv 'databricks_notebook.hello_world' 'module.databricks.databricks_notebook.hello_world'
terraform state mv 'databricks_notebook.kaggle_ingestion' 'module.databricks.databricks_notebook.kaggle_ingestion'
terraform state mv 'databricks_schema.bronze' 'module.databricks.databricks_schema.bronze'
terraform state mv 'databricks_schema.gold' 'module.databricks.databricks_schema.gold'
terraform state mv 'databricks_schema.silver' 'module.databricks.databricks_schema.silver'

echo "State migration complete!"
