import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from openai import OpenAI
from pydantic import BaseModel

from .config import LegalRagConfig
from .db import LegalDatabase
from .embeddings import EmbeddingService
from .ingestion import LegalCorpusIngestor
from .retrieval import HybridRetriever
from .s3_client import S3CorpusClient

logger = logging.getLogger(__name__)


class LegalIngestRequest(BaseModel):
    category: Optional[str] = None
    bucket_slug: Optional[str] = None
    manifest_path: Optional[str] = None


class LegalSearchRequest(BaseModel):
    query: str
    category: Optional[str] = None
    bucket_slug: Optional[str] = None
    year: Optional[int] = None
    limit: int = 10


class LegalAskRequest(LegalSearchRequest):
    top_k_context: int = 6


_svc_cache = None


def get_cached_db():
    """Return the LegalDatabase instance if already initialized, else None."""
    return _svc_cache[1] if _svc_cache is not None else None


def _services():
    global _svc_cache
    if _svc_cache is not None:
        return _svc_cache
    config = LegalRagConfig.from_env()
    config.validate_s3()
    config.validate_db()
    if not config.openai_api_key:
        raise ValueError("OPENAI_API_KEY is required")
    db = LegalDatabase(config.postgres_url)
    db.ensure_schema()
    s3 = S3CorpusClient(config.s3_bucket_name, config.aws_region)
    embeddings = EmbeddingService(config.openai_api_key, config.embedding_model, config.openai_base_url)
    retriever = HybridRetriever(db, embeddings)
    ingestor = LegalCorpusIngestor(db, s3, embeddings, config.s3_prefix)
    _svc_cache = (config, db, retriever, ingestor)
    return _svc_cache


router = APIRouter(prefix="/legal", tags=["legal-rag"])


@router.post("/ingest")
def ingest_legal_corpus(request: LegalIngestRequest):
    payload = request.model_dump()
    try:
        _, _, _, ingestor = _services()
    except Exception as exc:
        logger.exception(
            "POST /legal/ingest failed while initializing legal RAG (env, DB, S3, OpenAI) payload=%s",
            payload,
        )
        raise HTTPException(status_code=500, detail=str(exc) or repr(exc)) from exc
    try:
        return ingestor.ingest(
            category=request.category,
            bucket_slug=request.bucket_slug,
            manifest_path=request.manifest_path,
        )
    except FileNotFoundError as exc:
        logger.warning("POST /legal/ingest not found: %s payload=%s", exc, payload)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        logger.warning("POST /legal/ingest bad request: %s payload=%s", exc, payload)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.error(
            "POST /legal/ingest HTTP 500 payload=%s detail=%s (full traceback logged by legal_rag.ingestion if failure was inside ingest)",
            payload,
            exc,
        )
        raise HTTPException(status_code=500, detail=str(exc) or repr(exc)) from exc


@router.get("/manifests")
def list_legal_manifests(limit: int = 500, contains: Optional[str] = None):
    """List `.manifest.json` keys under `{LEGAL_S3_PREFIX}manifests/` for the configured bucket."""
    try:
        _, _, _, ingestor = _services()
        keys = ingestor.s3.list_manifest_keys(ingestor.s3_prefix)
        if contains:
            keys = [k for k in keys if contains in k]
        keys = keys[: max(1, min(limit, 5000))]
        return {
            "bucket": ingestor.s3.bucket_name,
            "aws_region": ingestor.s3.client.meta.region_name,
            "s3_prefix": ingestor.s3_prefix,
            "count": len(keys),
            "manifests": keys,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc) or repr(exc)) from exc


@router.get("/search")
def legal_search(
    query: str,
    category: Optional[str] = None,
    bucket_slug: Optional[str] = None,
    year: Optional[int] = None,
    limit: int = 10,
):
    try:
        _, _, retriever, _ = _services()
        rows = retriever.search(query=query, category=category, bucket_slug=bucket_slug, year=year, limit=limit)
        return {"query": query, "count": len(rows), "results": rows}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.post("/ask")
def legal_ask(request: LegalAskRequest):
    try:
        config, _, retriever, _ = _services()
        rows = retriever.search(
            query=request.query,
            category=request.category,
            bucket_slug=request.bucket_slug,
            year=request.year,
            limit=max(request.top_k_context, request.limit),
            include_full_text=True,
        )
        if not rows:
            return {"query": request.query, "answer": "No relevant legal documents found.", "citations": []}

        context_rows = rows[: request.top_k_context]
        context_blocks = []
        citations = []
        for idx, row in enumerate(context_rows, start=1):
            # full_text has data in ~63% of docs; summary is a short fallback; snippet is last resort
            body = (
                row.get("full_text")
                or row.get("summary")
                or row.get("snippet")
                or ""
            )
            # Cap at ~4000 chars so we don't blow up the context window on long statutes
            if len(body) > 4000:
                body = body[:4000] + "…"
            context_blocks.append(
                f"[{idx}] {row.get('title') or 'Untitled'} | {row.get('category')}/{row.get('bucket_slug')} | {row.get('year')}\n"
                f"{body}"
            )
            citations.append(
                {
                    "title": row.get("title"),
                    "bucket_slug": row.get("bucket_slug"),
                    "category": row.get("category"),
                    "year": row.get("year"),
                    "source_url": row.get("source_url"),
                    "s3_json_path": row.get("s3_json_path"),
                    "snippet": row.get("snippet"),
                    "full_text": row.get("full_text"),
                }
            )

        client = OpenAI(api_key=config.openai_api_key, base_url=config.openai_base_url) if config.openai_base_url else OpenAI(api_key=config.openai_api_key)
        prompt = (
            "You are a legal RAG assistant. Answer using only provided context. "
            "If uncertain, explicitly say so. Cite sources inline as [n].\n\n"
            f"Question: {request.query}\n\nContext:\n" + "\n\n".join(context_blocks)
        )
        completion = client.chat.completions.create(
            model=config.chat_model,
            messages=[
                {"role": "system", "content": "Provide concise, legally grounded answers with citations."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        answer = completion.choices[0].message.content
        return {"query": request.query, "answer": answer, "citations": citations, "results": rows[: request.limit]}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/buckets")
def list_legal_buckets():
    try:
        _, db, _, _ = _services()
        return {"buckets": db.list_buckets()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/documents/{document_id}")
def get_legal_document(document_id: int):
    try:
        _, db, _, _ = _services()
        doc = db.get_document(document_id)
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        return doc
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

