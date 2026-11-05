output "alb_dns_name" {
  description = "Public DNS name of the API load balancer."
  value       = aws_lb.api.dns_name
}

output "api_url" {
  description = "Base URL of the deployed API."
  value       = "http://${aws_lb.api.dns_name}"
}

output "ecr_repository_url" {
  description = "ECR repository URL to push the API image to."
  value       = aws_ecr_repository.api.repository_url
}

output "artifact_bucket" {
  description = "S3 bucket holding training data and model artifacts."
  value       = aws_s3_bucket.artifacts.bucket
}

output "ecs_cluster_name" {
  description = "ECS cluster running the API service."
  value       = aws_ecs_cluster.main.name
}
