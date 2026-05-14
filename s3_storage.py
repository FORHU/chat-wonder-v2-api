"""
AWS S3 Storage Module

Provides functions to upload, download, and manage files in AWS S3.
Falls back gracefully when S3 is not configured.
"""

import os
import logging
from typing import Optional
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


def get_s3_config() -> dict:
    bucket_name = os.getenv("LEGAL_S3_BUCKET_NAME", "")
    return {
        "use_s3": bool(bucket_name),
        "bucket_name": bucket_name,
        "region": os.getenv("AWS_REGION", "us-east-1"),
        "access_key": os.getenv("AWS_ACCESS_KEY_ID", ""),
        "secret_key": os.getenv("AWS_SECRET_ACCESS_KEY", ""),
    }


def get_s3_client():
    config = get_s3_config()
    if not config["use_s3"]:
        return None
    try:
        if config["access_key"] and config["secret_key"]:
            return boto3.client(
                "s3",
                aws_access_key_id=config["access_key"],
                aws_secret_access_key=config["secret_key"],
                region_name=config["region"],
            )
        return boto3.client("s3", region_name=config["region"])
    except NoCredentialsError:
        logger.error("AWS credentials not found.")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize S3 client: {e}")
        return None


def download_from_s3(s3_key: str, local_path: str) -> Optional[str]:
    config = get_s3_config()
    if not config["use_s3"]:
        return None
    s3_client = get_s3_client()
    if not s3_client:
        return None
    try:
        s3_client.download_file(config["bucket_name"], s3_key, local_path)
        return local_path
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code == "404":
            logger.warning(f"File not found in S3: {s3_key}")
        else:
            logger.error(f"S3 download error: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected S3 download error: {e}")
        return None


def upload_bytes_to_s3(data: bytes, s3_key: str, content_type: str = "application/octet-stream") -> bool:
    config = get_s3_config()
    if not config["use_s3"]:
        return False
    s3_client = get_s3_client()
    if not s3_client:
        return False
    try:
        s3_client.put_object(
            Bucket=config["bucket_name"],
            Key=s3_key,
            Body=data,
            ContentType=content_type,
        )
        return True
    except Exception as e:
        logger.error(f"S3 upload error: {e}")
        return False


def generate_presigned_put(
    s3_key: str,
    content_type: str = "application/octet-stream",
    expiration_in_seconds: int = 3600,
) -> Optional[str]:
    config = get_s3_config()
    if not config["use_s3"]:
        return None
    s3_client = get_s3_client()
    if not s3_client:
        return None
    try:
        return s3_client.generate_presigned_url(
            ClientMethod="put_object",
            Params={
                "Bucket": config["bucket_name"],
                "Key": s3_key,
                "ContentType": content_type,
            },
            ExpiresIn=expiration_in_seconds,
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned PUT URL: {e}")
        return None


def generate_presigned_get(
    s3_key: str, expiration_in_seconds: int = 604800
) -> Optional[str]:
    config = get_s3_config()
    if not config["use_s3"]:
        return None
    s3_client = get_s3_client()
    if not s3_client:
        return None
    try:
        return s3_client.generate_presigned_url(
            ClientMethod="get_object",
            Params={"Bucket": config["bucket_name"], "Key": s3_key},
            ExpiresIn=expiration_in_seconds,
        )
    except Exception as e:
        logger.error(f"Failed to generate presigned GET URL: {e}")
        return None
