import json
from typing import Iterable

import boto3
from botocore.exceptions import ClientError


class S3CorpusClient:
    def __init__(self, bucket_name: str, region_name: str):
        self.bucket_name = bucket_name
        self.client = boto3.client("s3", region_name=region_name)

    def list_keys(self, prefix: str) -> list[str]:
        keys: list[str] = []
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=prefix):
            for item in page.get("Contents", []):
                key = item.get("Key")
                if key:
                    keys.append(key)
        return keys

    def list_manifest_keys(self, base_prefix: str) -> list[str]:
        return [k for k in self.list_keys(f"{base_prefix}manifests/") if k.endswith(".manifest.json")]

    def read_json(self, key: str):
        uri = self.to_s3_uri(key)
        try:
            obj = self.client.get_object(Bucket=self.bucket_name, Key=key)
            body = obj["Body"].read().decode("utf-8")
            return json.loads(body)
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code", "")
            if code in ("404", "NoSuchKey", "NotFound"):
                raise FileNotFoundError(f"S3 object not found (check LEGAL_S3_BUCKET_NAME, LEGAL_S3_PREFIX, region, and key spelling): {uri}") from exc
            raise RuntimeError(f"S3 GetObject failed for {uri}: {exc}") from exc

    def object_exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket_name, Key=key)
            return True
        except Exception:
            return False

    def to_s3_uri(self, key: str) -> str:
        return f"s3://{self.bucket_name}/{key}"

    @staticmethod
    def first_existing_key(client: "S3CorpusClient", candidates: Iterable[str]):
        for key in candidates:
            if key and client.object_exists(key):
                return key
        return None

