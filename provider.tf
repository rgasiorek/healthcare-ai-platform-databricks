terraform {
  required_providers {
    databricks = {
      source  = "databricks/databricks"
      version = "~> 1.50"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  required_version = ">= 1.0"
}

provider "databricks" {
  # Configuration will be read from ~/.databrickscfg
}

provider "aws" {
  region  = "eu-central-1"
  profile = "DevAdmin-905418100642"

  default_tags {
    tags = {
      Project     = "healthcare-ai"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}
