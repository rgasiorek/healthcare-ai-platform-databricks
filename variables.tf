# Environment configuration
variable "environment" {
  description = "Environment name (dev, pilot, prod)"
  type        = string
  default     = "dev"
}

# AWS Region
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "eu-central-1"
}
