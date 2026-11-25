# -*- coding: utf-8 -*-
"""Liveness, readiness, version, and session bootstrap routes."""

from __future__ import annotations

import os
import time
import uuid

from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()


@router.get("/")
@router.get("/healthz")
async def liveness():
    """Liveness only: 200 whenever the process is up and the event loop is turning."""
    from the_server import _context

    return {
        "status": "ok",
        "service": "chat-wonder-v2",
        "sessions": len(_context.sessions),
    }


@router.get("/health")
async def health_check():
    """Readiness: 200 if healthy, 503 if any critical dependency is down."""
    import s3_storage
    from juris_mcp.client import get_client as get_juris_mcp_client
    from korean_law_mcp.client import get_client as get_korean_law_mcp_client
    from the_server import _context
    from uk_legal_mcp.client import get_client as get_uk_legal_mcp_client

    checks = {}
    overall = "healthy"

    try:
        t0 = time.monotonic()
        probe = get_juris_mcp_client().call_tool(
            "search_jurisprudence", {"query": "due process", "limit": 1}
        )
        ok = isinstance(probe, dict)
        checks["juris_mcp"] = {
            "status": "ok" if ok else "error",
            "url": _context.juris_mcp_url,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }
        if not ok:
            overall = "degraded"
    except Exception as e:
        checks["juris_mcp"] = {
            "status": "error",
            "url": _context.juris_mcp_url,
            "detail": str(e),
        }
        overall = "degraded"

    try:
        t0 = time.monotonic()
        probe = get_uk_legal_mcp_client().call_tool(
            "case_law_search", {"query": "negligence duty of care", "limit": 1}
        )
        ok = isinstance(probe, dict)
        checks["uk_legal_mcp"] = {
            "status": "ok" if ok else "error",
            "url": _context.uk_legal_mcp_url,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }
        if not ok:
            overall = "degraded"
    except Exception as e:
        checks["uk_legal_mcp"] = {
            "status": "error",
            "url": _context.uk_legal_mcp_url,
            "detail": str(e),
        }
        overall = "degraded"

    try:
        t0 = time.monotonic()
        probe = get_korean_law_mcp_client().call_tool(
            "search_law", {"query": "민법", "display": 1}
        )
        ok = isinstance(probe, dict)
        checks["korean_law_mcp"] = {
            "status": "ok" if ok else "error",
            "url": _context.korean_law_mcp_url,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }
        if not ok:
            overall = "degraded"
    except Exception as e:
        checks["korean_law_mcp"] = {
            "status": "error",
            "url": _context.korean_law_mcp_url,
            "detail": str(e),
        }
        overall = "degraded"

    try:
        legal_bucket = os.getenv("LEGAL_S3_BUCKET_NAME") or os.getenv("S3_BUCKET_NAME")
        if not legal_bucket:
            checks["s3"] = {"status": "unconfigured"}
        else:
            t0 = time.monotonic()
            s3 = s3_storage.get_s3_client()
            s3.head_bucket(Bucket=legal_bucket)
            checks["s3"] = {
                "status": "ok",
                "bucket": legal_bucket,
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }
    except Exception as e:
        checks["s3"] = {"status": "error", "detail": str(e)}
        overall = "degraded"

    try:
        cosmetics_bucket = os.getenv("COSMETICS_S3_BUCKET_NAME")
        cosmetics_region = os.getenv("COSMETICS_AWS_REGION")
        if not cosmetics_bucket:
            checks["s3_cosmetics"] = {"status": "unconfigured"}
        else:
            t0 = time.monotonic()
            s3 = s3_storage.get_s3_client(
                bucket_name=cosmetics_bucket, region=cosmetics_region
            )
            s3.head_bucket(Bucket=cosmetics_bucket)
            checks["s3_cosmetics"] = {
                "status": "ok",
                "bucket": cosmetics_bucket,
                "latency_ms": round((time.monotonic() - t0) * 1000),
            }
    except Exception as e:
        checks["s3_cosmetics"] = {
            "status": "error",
            "bucket": os.getenv("COSMETICS_S3_BUCKET_NAME"),
            "detail": str(e),
        }
        overall = "degraded"

    try:
        t0 = time.monotonic()
        probe = get_juris_mcp_client().call_tool(
            "search_jurisprudence", {"query": "due process", "limit": 1}
        )
        ok = isinstance(probe, dict)
        checks["juris_mcp"] = {
            "status": "ok" if ok else "error",
            "url": _context.juris_mcp_url,
            "latency_ms": round((time.monotonic() - t0) * 1000),
        }
        if not ok:
            overall = "degraded"
    except Exception as e:
        checks["juris_mcp"] = {
            "status": "error",
            "url": _context.juris_mcp_url,
            "detail": str(e),
        }
        overall = "degraded"

    if _context.openai_api_key:
        checks["openai"] = {"status": "configured", "model": _context.model}
    else:
        checks["openai"] = {"status": "unconfigured"}
        overall = "degraded"

    status_code = 200 if overall == "healthy" else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": overall, "checks": checks},
    )


@router.get("/config")
async def get_config():
    from the_server import _context

    return {
        "legal_library_url": _context.legal_library_url,
        "juris_mcp_url": _context.juris_mcp_url,
        "uk_legal_mcp_url": _context.uk_legal_mcp_url,
        "korean_law_mcp_url": _context.korean_law_mcp_url,
    }


@router.get("/version")
async def get_version():
    import confidence
    import model_registry
    from the_server import _context

    return {
        "service": "chat-wonder-v2-api",
        "build_marker": _context.build_marker,
        "chat_model": _context.model,
        "legal_source": "juris_mcp",
        "citation_guard": {
            "enabled": True,
            "mode": "juris_ph_urls",
            "metrics": [
                "legal.citation_repair.count",
            ],
        },
        "confidence": {
            "enabled": confidence.confidence_enabled(),
            "schema_version": confidence.SCHEMA_VERSION,
            "entropy_models": list(model_registry.logprob_capable_prefixes()),
            "verification_personas": ["legal", "legal_uk", "legal_kr"],
        },
    }


@router.get("/session-id")
async def get_new_session_id():
    from the_server import ChatState, _context

    new_id = str(uuid.uuid4())
    _context.sessions[new_id] = ChatState()
    return {"session_id": new_id}
