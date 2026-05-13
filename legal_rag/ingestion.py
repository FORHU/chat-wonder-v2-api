import hashlib
import logging
import os
from dataclasses import dataclass
from typing import Any


def _env_int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(minimum, value)


LEGAL_INGEST_MAX_INDEX_CHARS = _env_int("LEGAL_INGEST_MAX_INDEX_CHARS", 400_000, 5_000)
LEGAL_INGEST_MAX_CHUNKS_PER_DOC = _env_int("LEGAL_INGEST_MAX_CHUNKS_PER_DOC", 2048, 16)

from .chunking import chunk_legal_text, normalize_text
from .db import LegalDatabase
from .embeddings import EmbeddingService
from .s3_client import S3CorpusClient

logger = logging.getLogger(__name__)


@dataclass
class IngestionStats:
    manifests: int = 0
    json_files: int = 0
    documents_upserted: int = 0
    updated_changed: int = 0
    chunks_written: int = 0
    skipped_empty: int = 0
    skipped_unchanged: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "manifests": self.manifests,
            "json_files": self.json_files,
            "documents_upserted": self.documents_upserted,
            "updated_changed": self.updated_changed,
            "chunks_written": self.chunks_written,
            "skipped_empty": self.skipped_empty,
            "skipped_unchanged": self.skipped_unchanged,
        }


class LegalCorpusIngestor:
    def __init__(self, db: LegalDatabase, s3: S3CorpusClient, embeddings: EmbeddingService, s3_prefix: str):
        self.db = db
        self.s3 = s3
        self.embeddings = embeddings
        self.s3_prefix = s3_prefix

    def _normalize_manifest_key(self, manifest_path: str) -> str:
        key = (manifest_path or "").strip()
        if not key:
            raise ValueError("manifest_path is empty")
        if key.startswith("s3://"):
            remainder = key[5:]
            bucket, sep, rest = remainder.partition("/")
            if not sep:
                raise ValueError(f"Invalid S3 URI: {manifest_path!r}")
            if bucket != self.s3.bucket_name:
                raise ValueError(
                    f"S3 URI bucket {bucket!r} does not match LEGAL_S3_BUCKET_NAME ({self.s3.bucket_name!r})"
                )
            key = rest
        key = key.lstrip("/")
        if self.s3_prefix and not key.startswith(self.s3_prefix):
            key = f"{self.s3_prefix}{key.lstrip('/')}"
        return key

    def ingest(self, category=None, bucket_slug=None, manifest_path=None) -> dict[str, Any]:
        source_ref = manifest_path or f"{category or '*'}:{bucket_slug or '*'}"
        run_id = self.db.create_ingestion_run("s3-anycase", source_ref)
        stats = IngestionStats()
        try:
            logger.info(
                "Legal ingest started run_id=%s bucket=%s prefix=%r source_ref=%r category=%r bucket_slug=%r manifest_path=%r",
                run_id,
                self.s3.bucket_name,
                self.s3_prefix,
                source_ref,
                category,
                bucket_slug,
                manifest_path,
            )
            manifests = self._discover_manifests(category=category, bucket_slug=bucket_slug, manifest_path=manifest_path)
            stats.manifests = len(manifests)
            logger.info("Legal ingest discovered %s manifest(s): %s", len(manifests), manifests[:25])
            if len(manifests) > 25:
                logger.info("Legal ingest … (%s more manifest keys omitted)", len(manifests) - 25)
            for manifest_key in manifests:
                self._ingest_manifest(manifest_key, stats)
            self.db.complete_ingestion_run(run_id, "completed", stats.to_dict())
            logger.info(
                "Legal ingest completed run_id=%s stats=%s",
                run_id,
                stats.to_dict(),
            )
            return {"run_id": run_id, "status": "completed", "stats": stats.to_dict()}
        except Exception as exc:
            logger.exception(
                "Legal ingest failed run_id=%s bucket=%s stats_so_far=%s",
                run_id,
                self.s3.bucket_name,
                stats.to_dict(),
            )
            self.db.complete_ingestion_run(run_id, "failed", stats.to_dict(), error={"message": str(exc)})
            raise

    def _discover_manifests(self, category, bucket_slug, manifest_path) -> list[str]:
        if manifest_path:
            return [self._normalize_manifest_key(manifest_path)]
        manifests = self.s3.list_manifest_keys(self.s3_prefix)
        filtered: list[str] = []
        for key in manifests:
            parts = key.split("/")
            if len(parts) < 4:
                continue
            item_category = parts[-2]
            item_bucket = parts[-1].replace(".manifest.json", "")
            if category and category != item_category:
                continue
            if bucket_slug and bucket_slug != item_bucket:
                continue
            filtered.append(key)
        return filtered

    def _ingest_manifest(self, manifest_key: str, stats: IngestionStats) -> None:
        logger.info("Legal ingest manifest begin key=%s uri=%s", manifest_key, self.s3.to_s3_uri(manifest_key))
        manifest = self.s3.read_json(manifest_key)
        category, bucket_slug = self._extract_category_bucket(manifest_key)
        json_candidates = self._manifest_json_paths(manifest, category, bucket_slug)
        logger.info(
            "Legal ingest manifest resolved category=%s bucket_slug=%s corpus_json_keys=%s",
            category,
            bucket_slug,
            json_candidates,
        )
        if not json_candidates:
            logger.warning(
                "Legal ingest manifest has no corpus JSON keys after resolution manifest=%s",
                manifest_key,
            )
        list_json = [k for k in json_candidates if "full_details" not in k]
        full_json = [k for k in json_candidates if "full_details" in k]
        all_json = full_json + list_json
        logger.info(
            "Legal ingest loading %s corpus JSON file(s) order=full_details_first manifest=%s",
            len(all_json),
            manifest_key,
        )

        all_records: dict[str, dict[str, Any]] = {}
        for idx, json_key in enumerate(all_json, start=1):
            stats.json_files += 1
            logger.debug(
                "Legal ingest reading corpus JSON %s/%s key=%s",
                idx,
                len(all_json),
                json_key,
            )
            try:
                data = self.s3.read_json(json_key)
            except Exception:
                logger.exception(
                    "Legal ingest corpus JSON read failed manifest=%s key=%s uri=%s",
                    manifest_key,
                    json_key,
                    self.s3.to_s3_uri(json_key),
                )
                raise
            records = self._extract_records(data)
            is_full = "full_details" in json_key
            for record in records:
                record = dict(record)
                record["_s3_path"] = json_key
                doc_key = self._record_key(record, fallback=f"{json_key}:{len(all_records)}")
                merged = all_records.get(doc_key, {})
                all_records[doc_key] = self._merge_records(merged, record, prefer_new=is_full)

        logger.info(
            "Legal ingest manifest merged records manifest=%s unique_records=%s",
            manifest_key,
            len(all_records),
        )
        for record in all_records.values():
            self._upsert_record(record, category, bucket_slug, manifest_key, stats)
        logger.info("Legal ingest manifest done key=%s", manifest_key)

    def _extract_category_bucket(self, manifest_key: str) -> tuple[str, str]:
        parts = manifest_key.split("/")
        category = parts[-2]
        bucket_slug = parts[-1].replace(".manifest.json", "")
        return category, bucket_slug

    @staticmethod
    def _is_corpus_json_reference(path: str) -> bool:
        """Manifests often mention *.errors.json logs; those are not ingestion inputs."""
        name = path.split("/")[-1]
        return bool(name) and not name.endswith(".errors.json")

    def _resolve_corpus_json_key(self, raw: str, category: str, bucket_slug: str) -> str:
        """Turn manifest-relative paths into bucket keys. Bare filenames live under json/<category>/<bucket_slug>/."""
        key = raw.strip()
        if key.startswith("s3://"):
            _, _, remainder = key.partition("s3://")
            _bucket, sep, path = remainder.partition("/")
            key = path if sep else remainder
        key = key.lstrip("/")
        pref = self.s3_prefix

        if key.startswith(pref):
            return key
        if key.startswith("json/"):
            return f"{pref}{key}"
        if "/" not in key:
            primary = f"{pref}json/{category}/{bucket_slug}/{key}"
            secondary = f"{pref}{key}"
            found = S3CorpusClient.first_existing_key(self.s3, (primary, secondary))
            return found or primary
        return f"{pref}{key.lstrip('/')}"

    def _manifest_json_paths(self, manifest, category: str, bucket_slug: str) -> list[str]:
        candidates: list[str] = []

        def walk(value: Any):
            if isinstance(value, dict):
                for v in value.values():
                    walk(v)
            elif isinstance(value, list):
                for v in value:
                    walk(v)
            elif isinstance(value, str) and value.endswith(".json"):
                if self._is_corpus_json_reference(value):
                    candidates.append(value)

        walk(manifest)
        normalized = []
        for raw in candidates:
            resolved = self._resolve_corpus_json_key(raw, category, bucket_slug)
            if self._is_corpus_json_reference(resolved):
                normalized.append(resolved)

        fallback = [
            f"{self.s3_prefix}json/{category}/{bucket_slug}/anycase_{bucket_slug}_full_details.json",
            f"{self.s3_prefix}json/{category}/{bucket_slug}/anycase_{bucket_slug}.json",
        ]
        for item in fallback:
            if item not in normalized and self.s3.object_exists(item):
                normalized.append(item)
        return sorted(set(normalized))

    def _extract_records(self, payload) -> list[dict[str, Any]]:
        if isinstance(payload, list):
            return [row for row in payload if isinstance(row, dict)]
        if isinstance(payload, dict):
            for key in ("records", "items", "results", "data", "documents"):
                value = payload.get(key)
                if isinstance(value, list):
                    return [row for row in value if isinstance(row, dict)]
            return [payload]
        return []

    def _record_key(self, record: dict[str, Any], fallback: str) -> str:
        for key in ("id", "doc_id", "case_no", "case_number", "slug", "url"):
            value = record.get(key)
            if value:
                return str(value).strip().lower()
        title = normalize_text(record.get("title") or record.get("case_title") or "")
        year = str(record.get("year") or "")
        if title:
            return f"{title.lower()}::{year}"
        return fallback

    def _merge_records(self, old: dict[str, Any], new: dict[str, Any], prefer_new: bool) -> dict[str, Any]:
        if not old:
            return dict(new)
        merged = dict(old)
        for key, value in new.items():
            if prefer_new or not merged.get(key):
                merged[key] = value
        return merged

    def _upsert_record(self, record: dict[str, Any], category: str, bucket_slug: str, manifest_key: str, stats: IngestionStats) -> None:
        title = normalize_text(record.get("title") or record.get("case_title"))
        case_no = normalize_text(record.get("case_no") or record.get("case_number") or record.get("gr_number"))
        summary = normalize_text(record.get("summary"))
        concise_summary = normalize_text(record.get("concise_summary"))
        full_text = normalize_text(record.get("full_text") or record.get("content") or record.get("text"))
        full_text_source = "full_text" if full_text else ("concise_summary" if concise_summary else "summary")
        index_text = full_text or concise_summary or summary
        if len(index_text) < 60:
            stats.skipped_empty += 1
            return

        year_value = record.get("year")
        try:
            year = int(year_value) if year_value is not None else None
        except (ValueError, TypeError):
            year = None

        s3_json_path = record.get("_s3_path") or record.get("s3_json_path") or ""
        if not s3_json_path:
            for candidate in ("source_file", "json_source", "source_json"):
                if record.get(candidate):
                    s3_json_path = record[candidate]
                    break

        source_hash = hashlib.sha256(
            f"{category}|{bucket_slug}|{case_no}|{title}|{year}|{s3_json_path}".encode("utf-8")
        ).hexdigest()
        content_hash = hashlib.sha256(index_text.encode("utf-8")).hexdigest()

        existing = self.db.get_document_hashes(source_hash)
        if existing and existing.get("content_hash") == content_hash:
            stats.skipped_unchanged += 1
            return
        if existing:
            stats.updated_changed += 1

        doc_id = self.db.upsert_document(
            {
                "source_hash": source_hash,
                "content_hash": content_hash,
                "bucket_slug": bucket_slug,
                "category": category,
                "subcategory": record.get("subcategory"),
                "title": title or None,
                "case_no": case_no or None,
                "year": year,
                "source_url": record.get("source_url") or record.get("url"),
                "metadata_json": record.get("metadata") if isinstance(record.get("metadata"), dict) else record,
                "summary": summary or None,
                "concise_summary": concise_summary or None,
                "full_text": full_text or None,
                "full_text_source": full_text_source,
                "s3_json_path": s3_json_path or manifest_key.replace("manifests/", "json/"),
                "s3_manifest_path": manifest_key,
            }
        )
        stats.documents_upserted += 1

        raw_len = len(index_text)
        if raw_len > LEGAL_INGEST_MAX_INDEX_CHARS:
            logger.warning(
                "Legal ingest truncating index text for embeddings category=%s bucket_slug=%s title=%r chars=%s max=%s "
                "(full_text remains stored on document row; raise LEGAL_INGEST_MAX_INDEX_CHARS if needed)",
                category,
                bucket_slug,
                (title[:80] + "…") if title and len(title) > 80 else title,
                raw_len,
                LEGAL_INGEST_MAX_INDEX_CHARS,
            )
            index_text = index_text[:LEGAL_INGEST_MAX_INDEX_CHARS]

        chunks = chunk_legal_text(index_text, max_chunks=LEGAL_INGEST_MAX_CHUNKS_PER_DOC)
        if not chunks:
            stats.skipped_empty += 1
            return
        try:
            vectors = self.embeddings.embed_texts(chunks)
        except Exception:
            logger.exception(
                "Legal ingest embedding failed doc_id=%s chunks=%s category=%s bucket_slug=%s",
                doc_id,
                len(chunks),
                category,
                bucket_slug,
            )
            raise
        stats.chunks_written += self.db.replace_document_chunks(doc_id, chunks, vectors)

