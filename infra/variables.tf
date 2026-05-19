variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "ap-southeast-1"
}

variable "env" {
  description = "Environment label used in resource names"
  type        = string
  default     = "dev"
}

variable "aws_account_id" {
  description = "AWS account ID"
  type        = string
  default     = "617163942417"
}

variable "ecr_repository" {
  description = "ECR repository name"
  type        = string
  default     = "chat-wonder-v2-api"
}

variable "corpus_bucket_name" {
  description = "S3 bucket containing the legal corpus (read-only)"
  type        = string
  default     = "chat-wonder-dev"
}

variable "db_secret_arn" {
  description = "Secrets Manager ARN for PostgreSQL credentials"
  type        = string
  default     = "arn:aws:secretsmanager:ap-southeast-1:617163942417:secret:legal-rag/dev/postgres-64Un6h"
}

variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "t3.medium"
}

variable "key_name" {
  description = "EC2 key pair name"
  type        = string
  default     = "joel-key-pair"
}

variable "ssh_cidr_blocks" {
  description = "CIDR blocks allowed to SSH into the instance"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}
