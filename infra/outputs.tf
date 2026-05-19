output "public_ip" {
  description = "Elastic IP — use this as EC2_HOST in GitHub secrets and as the DNS A record for your domain"
  value       = aws_eip.app.public_ip
}

output "instance_id" {
  description = "EC2 instance ID"
  value       = aws_instance.app.id
}

output "iam_role_name" {
  description = "IAM role attached to the instance"
  value       = aws_iam_role.app.name
}
