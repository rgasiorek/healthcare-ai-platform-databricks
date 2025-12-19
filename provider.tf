terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.50"
    }
  }
  required_version = ">= 1.0"
}

provider "databricks" {
  # Configuration will be read from ~/.databrickscfg
}
