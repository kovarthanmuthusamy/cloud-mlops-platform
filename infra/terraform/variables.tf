variable "project_name" {
  description = "Name prefix for all resources."
  type        = string
  default     = "cloud-mlops-platform"
}

variable "environment" {
  description = "Deployment environment name."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = "AWS region to deploy into."
  type        = string
  default     = "eu-central-1"
}

variable "image_tag" {
  description = "Tag of the API container image in ECR to deploy."
  type        = string
  default     = "latest"
}

variable "container_port" {
  description = "Port the API container listens on."
  type        = number
  default     = 8000
}

variable "task_cpu" {
  description = "Fargate task CPU units (256, 512, 1024, ...)."
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate task memory in MiB."
  type        = number
  default     = 1024
}

variable "desired_count" {
  description = "Number of API tasks the ECS service should run."
  type        = number
  default     = 1
}
