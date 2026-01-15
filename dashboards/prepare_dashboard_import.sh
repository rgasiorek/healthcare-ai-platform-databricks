#!/bin/bash
# Prepare dashboard for import by injecting warehouse ID from Terraform

set -e

echo "Preparing dashboard for import..."

# Get warehouse ID from Terraform output
cd "$(dirname "$0")/../terraform"
WAREHOUSE_ID=$(terraform output -raw warehouse_id 2>/dev/null || echo "")

if [ -z "$WAREHOUSE_ID" ]; then
    echo "Error: Could not get warehouse_id from Terraform output"
    echo "Make sure you've run: terraform apply"
    exit 1
fi

echo "Found warehouse ID: $WAREHOUSE_ID"

# Update the dashboard JSON file
cd "$(dirname "$0")"
DASHBOARD_FILE="model_performance_dashboard_import.lvdash.json"

# Check if file exists
if [ ! -f "$DASHBOARD_FILE" ]; then
    echo "Error: Dashboard file not found: $DASHBOARD_FILE"
    exit 1
fi

# Replace placeholder with actual warehouse ID
if grep -q "YOUR_WAREHOUSE_ID" "$DASHBOARD_FILE"; then
    sed -i.bak "s/YOUR_WAREHOUSE_ID/$WAREHOUSE_ID/" "$DASHBOARD_FILE"
    echo "Updated $DASHBOARD_FILE with warehouse ID"
    echo ""
    echo "Dashboard ready for import!"
    echo ""
    echo "Next steps:"
    echo "1. Open Databricks workspace"
    echo "2. Go to Workspace → Create → Dashboard"
    echo "3. Click 'Import dashboard'"
    echo "4. Upload: $(pwd)/$DASHBOARD_FILE"
    echo "5. Click 'Import'"
else
    echo "Dashboard file already configured (no placeholder found)"
    echo "Current warehouse_id in file: $(grep '"warehouse_id"' "$DASHBOARD_FILE" | cut -d'"' -f4)"
fi

# Show file location
echo ""
echo "Dashboard file location:"
echo "$(pwd)/$DASHBOARD_FILE"
