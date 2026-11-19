#!/bin/sh
# Runs inside the LocalStack container on every start (mounted into /etc/localstack/init/ready.d).
# LocalStack keeps buckets in memory only, so the bucket is gone after a container restart -- this
# recreates it every time. Mirrors ilovelawyer-api/docker/localstack-init/ready-create-bucket.sh:
# public GetObject so local presigned/direct GETs behave like a real bucket, and a CORS rule so a
# browser can PUT straight to a presigned URL from generate_document_upload_url (see s3_storage.py's
# AWS_S3_ENDPOINT override).
set -eu

REGION="${DEFAULT_REGION:-ap-southeast-1}"
BUCKET="${AWS_S3_BUCKET:-chat-wonder-dev}"

awslocal s3 mb --region "$REGION" "s3://$BUCKET" 2>/dev/null || true
awslocal s3api put-bucket-policy --bucket "$BUCKET" --policy "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Effect\": \"Allow\",
    \"Principal\": \"*\",
    \"Action\": \"s3:GetObject\",
    \"Resource\": \"arn:aws:s3:::$BUCKET/*\"
  }]
}"
awslocal s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration "{
  \"CORSRules\": [{
    \"AllowedOrigins\": [\"*\"],
    \"AllowedMethods\": [\"GET\", \"PUT\", \"HEAD\", \"POST\"],
    \"AllowedHeaders\": [\"*\"],
    \"ExposeHeaders\": [\"ETag\"],
    \"MaxAgeSeconds\": 3000
  }]
}"
