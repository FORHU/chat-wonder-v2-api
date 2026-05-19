# EC2 IAM instance role for AWS access

The app container needs S3 read access (legal corpus) and the EC2 host needs ECR pull access. We use an IAM instance role attached to the EC2 rather than static access keys. Static keys would require two extra GitHub secrets (`APP_AWS_ACCESS_KEY_ID`, `APP_AWS_SECRET_ACCESS_KEY`) and still wouldn't solve ECR auth on the EC2 host — you'd need a role anyway. With the instance role, boto3 falls back to instance metadata credentials automatically (see `s3_storage.py`) and the AWS CLI on the host inherits the same role. No credentials are written to `user_functions.env` or stored in GitHub.

## Consequences

The EC2 must have the role attached at provisioning time. A bare EC2 without the role will fail at the ECR login step on first deploy.
