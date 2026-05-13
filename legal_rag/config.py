import os
from dataclasses import dataclass
from typing import Optional


def _normalize_prefix(value: str) -> str:
    prefix = (value or "anycase/").strip().strip("/")
    return f"{prefix}/"


@dataclass
class LegalRagConfig:
    aws_region: str
    s3_bucket_name: str
    s3_prefix: str
    postgres_url: str
    embedding_model: str
    chat_model: str
    openai_api_key: str
    openai_base_url: Optional[str]

    @classmethod
    def from_env(cls) -> "LegalRagConfig":
        postgres_url = os.getenv("LEGAL_DATABASE_URL", "")
        openai_api_key = os.getenv("OPENAI_API_KEY", "")
        return cls(
            aws_region=os.getenv("AWS_REGION", "us-east-1"),
            s3_bucket_name=os.getenv("LEGAL_S3_BUCKET_NAME", ""),
            s3_prefix=_normalize_prefix(os.getenv("LEGAL_S3_PREFIX", "anycase/")),
            postgres_url=postgres_url,
            embedding_model=os.getenv("LEGAL_EMBEDDING_MODEL", "text-embedding-3-small"),
            chat_model=os.getenv("LEGAL_CHAT_MODEL", "gpt-4o-mini"),
            openai_api_key=openai_api_key,
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
        )

    def validate_s3(self) -> None:
        if not self.s3_bucket_name:
            raise ValueError("LEGAL_S3_BUCKET_NAME is required")

    def validate_db(self) -> None:
        if not self.postgres_url:
            raise ValueError("LEGAL_DATABASE_URL is required")

