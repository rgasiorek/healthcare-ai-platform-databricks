# IAM Setup for Unity Catalog

## Issue
Your `PowerUserAccess` role doesn't have `iam:CreateRole` permission, which is required to create the IAM roles for Unity Catalog.

## Solution Options

### Option 1: Ask AWS Admin to Apply IAM Terraform (Recommended)

Your AWS admin needs to run:

```bash
# Login with admin credentials
aws sso login --profile <ADMIN_PROFILE>

# Apply only the IAM resources
terraform apply \
  -target=aws_iam_role.unity_catalog_metastore \
  -target=aws_iam_role.healthcare_data_access \
  -target=aws_iam_role_policy.unity_catalog_metastore_s3 \
  -target=aws_iam_role_policy.healthcare_data_s3

# The IAM config is in: aws_iam.tf.admin
# Temporarily rename it: mv aws_iam.tf.admin aws_iam.tf
# Then run the command above
# Then rename back: mv aws_iam.tf aws_iam.tf.admin
```

### Option 2: Manually Create IAM Roles in AWS Console

#### Role 1: Unity Catalog Metastore Role

**Name**: `databricks-unity-catalog-metastore-role`

**Trust Relationship**:
```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "AWS": "arn:aws:iam::414351767826:role/unity-catalog-prod-UCMasterRole-14S5ZJVKOTYTL"
    },
    "Action": "sts:AssumeRole",
    "Condition": {
      "StringEquals": {
        "sts:ExternalId": "5ed6b530-dbbf-4911-99df-b16b7863f1ef"
      }
    }
  }]
}
```

**Inline Policy** (name: `unity-catalog-metastore-s3-access`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation",
        "s3:GetLifecycleConfiguration",
        "s3:PutLifecycleConfiguration"
      ],
      "Resource": [
        "arn:aws:s3:::healthcare-databricks-unity-catalog-905418100642",
        "arn:aws:s3:::healthcare-databricks-unity-catalog-905418100642/*"
      ]
    }
  ]
}
```

#### Role 2: Healthcare Data Access Role

**Name**: `databricks-healthcare-data-access-role`

**Trust Relationship**: Same as Role 1

**Inline Policy** (name: `healthcare-data-s3-access`):
```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket",
        "s3:GetBucketLocation"
      ],
      "Resource": [
        "arn:aws:s3:::healthcare-data-lake-905418100642",
        "arn:aws:s3:::healthcare-data-lake-905418100642/*"
      ]
    }
  ]
}
```

---

## After IAM Roles Are Created

Once the IAM roles exist, you can continue with:

```bash
terraform apply
```

This will create the Unity Catalog resources that depend on those IAM roles.

---

## Role ARNs Needed

After creation, note these ARNs (you'll need them):
- Metastore Role: `arn:aws:iam::905418100642:role/databricks-unity-catalog-metastore-role`
- Data Access Role: `arn:aws:iam::905418100642:role/databricks-healthcare-data-access-role`
