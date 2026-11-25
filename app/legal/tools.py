"""Legal tool implementations extracted from user_functions."""
import os
import io
import json
import time
import uuid
import hashlib
import tempfile
import logging
import urllib.request
import urllib.parse
import urllib.error
from datetime import datetime, date
from llm_usage import make_openai_client

_search_cache = {}
_logger = logging.getLogger(__name__)
CACHE_TTL_SECONDS = 30

def _juris_call(tool_name: str, arguments: dict) -> dict:
    from juris_mcp.client import get_client

    t0 = time.perf_counter()
    payload = get_client().call_tool(tool_name, arguments)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _logger.info("juris_mcp %s elapsed=%.0fms", tool_name, elapsed_ms)
    return payload if isinstance(payload, dict) else {"data": payload}


def _normalize_search_results(rows: list, doc_kind: str) -> list:
    mapped = []
    for row in rows or []:
        if not isinstance(row, dict):
            continue
        url = row.get("url") or row.get("page_url") or ""
        title = (
            row.get("case_title")
            or row.get("title")
            or row.get("short_title")
            or row.get("ra_number")
            or ""
        )
        mapped.append(
            {
                "id": str(row.get("id") or ""),
                "item_id": str(row.get("id") or ""),
                "score": float(row.get("score", 0.0) or 0.0),
                "type": doc_kind,
                "title": title,
                "url": url,
                "case_number": row.get("case_number") or "",
                "ra_number": row.get("ra_number") or row.get("number") or "",
                "year": row.get("year"),
                "snippet": row.get("facts") or row.get("summary") or row.get("snippet") or "",
                "disposition": row.get("disposition") or "",
                "pdf_url": row.get("pdf_url") or row.get("source_pdf") or "",
                "metadata": row,
            }
        )
    return mapped


def search_jurisprudence(
    query: str = None,
    limit: int = 5,
    year: int = None,
    case_type: str = None,
) -> dict:
    """Semantic search over Philippine Supreme Court decisions via juris.ph MCP."""
    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "query is required"}

    limit = max(1, min(int(limit or 5), 20))
    cache_key = hashlib.md5(
        f"juris:{query}:{limit}:{year}:{case_type}".encode()
    ).hexdigest()
    if cache_key in _search_cache:
        cached = _search_cache[cache_key]
        if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
            out = cached["result"].copy()
            out["cached"] = True
            return out
        del _search_cache[cache_key]

    args = {"query": query, "limit": limit}
    if year is not None:
        args["year"] = int(year)
    if case_type:
        args["case_type"] = str(case_type)

    try:
        payload = _juris_call("search_jurisprudence", args)
        results = _normalize_search_results(payload.get("results") or [], "jurisprudence")
        result = {
            "success": True,
            "query": query,
            "limit": limit,
            "total_results": len(results),
            "results": results,
            "note": payload.get("note"),
            "search_type": "juris_mcp",
            "cached": False,
        }
        _search_cache[cache_key] = {"timestamp": time.time(), "result": result.copy()}
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Jurisprudence search failed: {e}",
        }


def search_republic_acts(query: str = None, limit: int = 5, year: int = None) -> dict:
    """Semantic search over Philippine Republic Acts via juris.ph MCP.

    Acronyms ("VAWC", "IPRA") and RA-number formats ("RA 9262" vs "R.A. No. 9262" vs "9262")
    are expanded before the query reaches juris.ph — see ph_legal_query.plan_ra_query and
    FORHU/chat-wonder-v2-api#75. The cache key is built from the expanded query set, not the
    raw input, so two spellings that expand to the same set (e.g. "vawc" vs "VAWC") share a
    cache entry.
    """
    from ph_legal_query import RaSearchPlan, document_number, plan_ra_query

    query = (query or "").strip()
    if not query:
        return {"success": False, "error": "query is required"}

    limit = max(1, min(int(limit or 5), 20))
    plan: RaSearchPlan = plan_ra_query(query)

    if plan.kind == "unindexed":
        return {
            "success": True,
            "query": query,
            "limit": limit,
            "total_results": 0,
            "results": [],
            "note": (
                f"{plan.label} isn't searchable here yet — juris.ph only indexes Republic Acts "
                "and Supreme Court decisions, not Executive Orders, Presidential Decrees, "
                "Administrative Orders or Memorandum Orders/Circulars. Try searching by topic, "
                "or by the Republic Act that amends it, if any."
            ),
            "search_type": "unindexed",
            "cached": False,
        }

    cache_key = hashlib.md5(f"ra:{'|'.join(plan.queries)}:{limit}:{year}".encode()).hexdigest()
    if cache_key in _search_cache:
        cached = _search_cache[cache_key]
        if time.time() - cached["timestamp"] < CACHE_TTL_SECONDS:
            out = cached["result"].copy()
            out["cached"] = True
            return out
        del _search_cache[cache_key]

    def _run(q: str) -> list:
        args = {"query": q, "limit": limit}
        if year is not None:
            args["year"] = int(year)
        payload = _juris_call("search_republic_acts", args)
        return _normalize_search_results(payload.get("results") or [], "republic_act")

    try:
        if len(plan.queries) == 1:
            results = _run(plan.queries[0])
        else:
            from concurrent.futures import ThreadPoolExecutor

            with ThreadPoolExecutor(max_workers=len(plan.queries)) as pool:
                batches = [f.result() for f in [pool.submit(_run, q) for q in plan.queries]]
            results = _merge_ra_results(batches, plan.ra_number, limit)

        result = {
            "success": True,
            "query": query,
            "limit": limit,
            "total_results": len(results),
            "results": results,
            "note": None,
            "search_type": "juris_mcp",
            "cached": False,
        }
        _search_cache[cache_key] = {"timestamp": time.time(), "result": result.copy()}
        return result
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"Republic Act search failed: {e}",
        }


def _merge_ra_results(batches: list, ra_number, limit: int) -> list:
    """Combines several search_republic_acts query variants into one list, deduplicated by id.
    The act with the exact RA number goes first — semantic search can rank it well behind
    amending acts that merely cite it — otherwise the best relevance score wins."""
    from ph_legal_query import document_number

    seen: set = set()
    merged = []
    for batch in batches:
        for item in batch:
            if item["id"] in seen:
                continue
            seen.add(item["id"])
            merged.append(item)

    if ra_number:
        exact = [it for it in merged if document_number(it.get("ra_number")) == ra_number]
        rest = [it for it in merged if document_number(it.get("ra_number")) != ra_number]
        return (exact + rest)[:limit]
    return sorted(merged, key=lambda it: it.get("score", 0.0), reverse=True)[:limit]


def get_case(
    id: str = None,
    case_number: str = None,
    include_full_text: bool = False,
) -> dict:
    """Fetch one Supreme Court decision by juris.ph id or case number (e.g. G.R. No. 254972)."""
    args = {"include_full_text": bool(include_full_text)}
    if id:
        args["id"] = str(id).strip()
    if case_number:
        args["case_number"] = str(case_number).strip()
    if not args.get("id") and not args.get("case_number"):
        return {"success": False, "error": "Provide id or case_number"}

    try:
        payload = _juris_call("get_case", args)
        if not payload:
            return {"success": False, "error": "Case not found"}
        url = payload.get("url") or payload.get("page_url") or ""
        title = payload.get("case_title") or payload.get("title") or ""
        return {
            "success": True,
            "type": "jurisprudence",
            "id": str(payload.get("id") or args.get("id") or ""),
            "item_id": str(payload.get("id") or args.get("id") or ""),
            "title": title,
            "url": url,
            "case_number": payload.get("case_number") or args.get("case_number") or "",
            "year": payload.get("year"),
            "pdf_url": payload.get("pdf_url") or payload.get("source_pdf") or "",
            "document": payload,
            "include_full_text": bool(include_full_text),
        }
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"get_case failed: {e}"}


def get_republic_act(
    id: str = None,
    ra_number: str = None,
    include_full_text: bool = False,
) -> dict:
    """Fetch one Republic Act by juris.ph id or RA number (e.g. RA 11313)."""
    args = {"include_full_text": bool(include_full_text)}
    if id:
        args["id"] = str(id).strip()
    if ra_number:
        args["ra_number"] = str(ra_number).strip()
    if not args.get("id") and not args.get("ra_number"):
        return {"success": False, "error": "Provide id or ra_number"}

    try:
        payload = _juris_call("get_republic_act", args)
        if not payload:
            return {"success": False, "error": "Republic Act not found"}
        url = payload.get("url") or payload.get("page_url") or ""
        title = payload.get("title") or payload.get("short_title") or payload.get("ra_number") or ""
        return {
            "success": True,
            "type": "republic_act",
            "id": str(payload.get("id") or args.get("id") or ""),
            "item_id": str(payload.get("id") or args.get("id") or ""),
            "title": title,
            "url": url,
            "ra_number": payload.get("ra_number") or args.get("ra_number") or "",
            "year": payload.get("year"),
            "pdf_url": payload.get("pdf_url") or payload.get("source_pdf") or "",
            "document": payload,
            "include_full_text": bool(include_full_text),
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "message": f"get_republic_act failed: {e}",
        }


def get_case_document(
    case_document_id: str,
    case_document_chunk_ids: list = None,
    offset: int = 0,
    limit: int = None,
    query: str = None,
) -> dict:
    """Fetch a user's own uploaded case document (contract, pleading, etc.) from
    ilovelawyer-api by id. Private to the uploading user — never treat this content
    as public jurisprudence or a Legal Citation.

    offset/limit page through the document's chunks (ordered by chunkIndex) when it's too
    large to read in one call — e.g. a 300-page exhibit fetched in full (no
    case_document_chunk_ids filter). Omit both for the normal case: either the relevance-
    filtered chunk set, or a document small enough to return whole. When has_more is true,
    call again with offset=next_offset to continue reading the same document.

    query is injected by execute_function_call from state.current_query, not something the
    model passes itself. Used only on the whole-document path (no case_document_chunk_ids,
    no limit) when the document's full text exceeds _CASE_DOCUMENT_CHAR_CAP: chunks are
    BM25-ranked against query before truncation, so the cut falls on the least relevant
    material instead of whatever sits last by chunkIndex. A document that fits under the cap
    is left in natural chunkIndex order regardless — nothing is being dropped, so there's no
    reason to disturb reading order. The offset/limit paging path is untouched (it relies on
    stable chunkIndex order across calls to `call again with offset=next_offset`)."""
    if not case_document_id:
        return {"success": False, "error": "Provide case_document_id"}

    base = (os.getenv("ILOVELAWYER_API_BASE") or "").rstrip("/")
    api_key = os.getenv("CHAT_WONDER_API_KEY", "")
    if not base:
        _logger.error("get_case_document id=%s missing_config=ILOVELAWYER_API_BASE", case_document_id)
        return {
            "success": False,
            "error": "missing_config",
            "message": "ILOVELAWYER_API_BASE is not set in user_functions.env",
        }
    if not api_key:
        _logger.error("get_case_document id=%s missing_config=CHAT_WONDER_API_KEY", case_document_id)
        return {
            "success": False,
            "error": "missing_config",
            "message": "CHAT_WONDER_API_KEY is not set in user_functions.env",
        }

    url = f"{base}/api/v1/case-document/{urllib.parse.quote(str(case_document_id))}"
    t0 = time.perf_counter()
    try:
        data = _http_get_json(url, {"x-api-key": api_key})
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        if e.code == 404:
            _logger.warning("get_case_document id=%s not_found elapsed=%.0fms", case_document_id, elapsed_ms)
            return {"success": False, "error": "not_found", "message": "Case document not found."}
        _logger.error("get_case_document id=%s http_%s elapsed=%.0fms error=%s", case_document_id, e.code, elapsed_ms, e)
        return {"success": False, "error": f"http_{e.code}", "message": f"get_case_document failed: {e}"}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.error("get_case_document id=%s failed elapsed=%.0fms error=%s", case_document_id, elapsed_ms, e)
        return {"success": False, "error": str(e), "message": f"get_case_document failed: {e}"}
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _logger.info("get_case_document id=%s fetched elapsed=%.0fms", case_document_id, elapsed_ms)

    name = data.get("name") or ""
    rag_status = data.get("ragStatus") or ""

    raw_chunks = data.get("chunks") or []
    chunks = raw_chunks
    _bm25_ordered = False
    if case_document_chunk_ids:
        wanted = set(case_document_chunk_ids)
        chunks = [c for c in raw_chunks if c.get("id") in wanted]
    elif query and limit is None and raw_chunks:
        _raw_total_chars = sum(len(c.get("chunkText", "") or "") for c in raw_chunks)
        if _raw_total_chars > _CASE_DOCUMENT_CHAR_CAP:
            from bm25 import rank as _bm25_rank

            _texts = [c.get("chunkText", "") or "" for c in raw_chunks]
            _order = _bm25_rank(_texts, query)
            chunks = [raw_chunks[i] for i in _order]
            _bm25_ordered = True

    if not _bm25_ordered:
        chunks = sorted(chunks, key=lambda c: c.get("chunkIndex", 0))
    total_chunks = len(chunks)
    has_more = False
    next_offset = None
    if limit is not None:
        _offset = max(0, int(offset or 0))
        _limit = max(1, int(limit))
        page = chunks[_offset:_offset + _limit]
        has_more = _offset + _limit < total_chunks
        next_offset = _offset + _limit if has_more else None
        chunks = page
    text = "\n\n".join(c.get("chunkText", "") for c in chunks).strip()

    if not text:
        if rag_status == "FAILED":
            _empty_message = "Text extraction failed for this document — no content is available."
        elif rag_status == "PENDING":
            _empty_message = "This document is still being processed — no content is available yet."
        elif raw_chunks and case_document_chunk_ids:
            # Document has real content, but none of its chunks made the relevance-filtered
            # id list — happens routinely with multiple active documents where only some are
            # relevant to the current question. Distinct from FAILED/PENDING, not an error.
            _empty_message = "No chunks relevant to the current question were found in this document."
        else:
            _empty_message = "No extracted content available yet — the document may still be processing."
        _logger.info("get_case_document id=%s empty ragStatus=%s", case_document_id, rag_status)
        return {
            "success": True,
            "id": case_document_id,
            "name": name,
            "chunks_found": 0,
            "message": _empty_message,
        }

    truncated = False
    if len(text) > _CASE_DOCUMENT_CHAR_CAP:
        text = text[:_CASE_DOCUMENT_CHAR_CAP].rstrip() + "\n\n[...truncated...]"
        truncated = True

    # A relevance-filtered, paged or char-capped read is NOT the document. Say so explicitly:
    # confirmed on the Brackenmoor benchmark that the model otherwise treats the filtered slice
    # as the whole exhibit ("D01 ... contains paragraphs 1-21 but not paragraph 25") and never
    # re-fetches — losing paragraphs the question cited by number.
    document_chunk_count = len(raw_chunks)
    partial = bool(case_document_chunk_ids and len(chunks) < document_chunk_count) or has_more or truncated
    result = {
        "success": True,
        "id": case_document_id,
        "name": name,
        "text": text,
        "chunk_count": len(chunks),
        "total_chunks": total_chunks,
        "document_chunk_count": document_chunk_count,
        "partial": partial,
        "has_more": has_more,
        "next_offset": next_offset,
    }
    if partial:
        if has_more:
            how = f"call again with offset={next_offset} to continue"
        elif case_document_chunk_ids:
            how = "call again WITHOUT case_document_chunk_ids to read the whole document"
        else:
            how = "call again with offset/limit to page through the rest"
        result["note"] = (
            f"PARTIAL: {len(chunks)} of {document_chunk_count} chunks returned. Do not treat this as "
            f"the complete document — if the question refers to a paragraph, part or item you cannot "
            f"see here, {how}."
        )
    _logger.info(
        "get_case_document id=%s chunks=%d/%d partial=%s truncated=%s has_more=%s",
        case_document_id, len(chunks), document_chunk_count, partial, truncated, has_more,
    )
    return result


# ---------------------------------------------------------------------------
# UK Legal MCP (uk-legal-mcp.fly.dev) — [legal ai uk] persona. See ADR-0003.
# ---------------------------------------------------------------------------

def _uk_tool_failure(payload) -> str:
    """The error text when a tool call came back flagged `isError`, else ''.

    The server answers HTTP 200 and the unwrap in mcp_client hands the error text back as an ordinary result
    ({"text": "Internal error: ...", "raw_result": {"isError": true}}), so without this check a failed search
    was reported to the model — and the trace — as `success: true`."""
    if not isinstance(payload, dict):
        return ""
    raw = payload.get("raw_result")
    if isinstance(raw, dict) and raw.get("isError"):
        return str(payload.get("text") or "the UK Legal MCP reported an error")[:500]
    return ""


def _uk_call(tool_name: str, arguments: dict) -> dict:
    """Call one UK Legal MCP tool and unwrap its JSON-RPC result. Raises UkLegalMcpToolError if the tool failed."""
    from uk_legal_mcp.client import UkLegalMcpToolError, get_client

    t0 = time.perf_counter()
    payload = get_client().call_tool(tool_name, arguments)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _logger.info("uk_legal_mcp %s elapsed=%.0fms", tool_name, elapsed_ms)
    failure = _uk_tool_failure(payload)
    if failure:
        raise UkLegalMcpToolError(failure)
    return payload if isinstance(payload, dict) else {"data": payload}


def _uk_direct_fallback_enabled() -> bool:
    return os.getenv("UK_LEGISLATION_DIRECT_FALLBACK", "true").strip().lower() not in ("false", "0", "no")


def _uk_with_direct_fallback(tool_name: str, arguments: dict, direct_call) -> dict:
    """The UK Legal MCP first; legislation.gov.uk itself if the MCP fails.

    uk-legal-mcp.fly.dev cannot currently get past legislation.gov.uk's AWS WAF challenge, while our own hosts
    can (see uk_legal_mcp/direct.py). `direct_call` receives that module and returns the same shape as the MCP.
    Both failing raises one error naming both reasons. UK_LEGISLATION_DIRECT_FALLBACK=false turns it off."""
    try:
        return _uk_call(tool_name, arguments)
    except Exception as mcp_exc:
        if not _uk_direct_fallback_enabled():
            raise
        _logger.warning("uk_legal_mcp %s failed (%s); trying legislation.gov.uk directly", tool_name, mcp_exc)
        from uk_legal_mcp import direct as uk_direct

        t0 = time.perf_counter()
        try:
            result = direct_call(uk_direct)
        except Exception as direct_exc:
            raise RuntimeError(f"{mcp_exc}; direct legislation.gov.uk lookup also failed: {direct_exc}") from direct_exc
        _logger.info("legislation_direct %s elapsed=%.0fms", tool_name, (time.perf_counter() - t0) * 1000)
        return {**result, "fallback": uk_direct.SOURCE, "mcp_error": str(mcp_exc)[:300]}


def _uk_result_rows(payload: dict):
    """First list of result rows under a common container key, or None.

    uk-legal-mcp.fly.dev's exact payload shape is an external server we don't
    control (see the_server.py's _trace_count_for_result), so probe the same
    common keys rather than assuming one fixed shape.
    """
    for key in ("results", "items", "matches", "sections"):
        val = payload.get(key)
        if isinstance(val, list):
            return val
    return None


def case_law_search(query: str = None, court: str = None, party: str = None, judge: str = None, from_date: str = None, to_date: str = None, limit: int = 10, page: int = 1) -> dict:
    """Search UK case law via the UK Legal MCP."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return {"success": False, "error": "query is required"}
    args = {}
    args["query"] = query
    if court is not None:
        args["court"] = str(court)
    if party is not None:
        args["party"] = str(party)
    if judge is not None:
        args["judge"] = str(judge)
    if from_date is not None:
        args["from_date"] = str(from_date)
    if to_date is not None:
        args["to_date"] = str(to_date)
    if limit is not None:
        args["limit"] = int(limit)
    if page is not None:
        args["page"] = int(page)
    try:
        payload = _uk_call("case_law_search", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"case_law_search failed: {e}"}


def judgment_get_header(slug: str = None) -> dict:
    """Fetch a UK judgment's header metadata by slug."""
    slug = (str(slug).strip() if slug is not None else "")
    if not slug:
        return {"success": False, "error": "slug is required"}
    args = {}
    args["slug"] = slug
    try:
        payload = _uk_call("judgment_get_header", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"judgment_get_header failed: {e}"}


def judgment_get_index(slug: str = None) -> dict:
    """Fetch a UK judgment's paragraph index by slug."""
    slug = (str(slug).strip() if slug is not None else "")
    if not slug:
        return {"success": False, "error": "slug is required"}
    args = {}
    args["slug"] = slug
    try:
        payload = _uk_call("judgment_get_index", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"judgment_get_index failed: {e}"}


def judgment_get_paragraph(slug: str = None, eId: str = None) -> dict:
    """Fetch one paragraph of a UK judgment by slug + eId."""
    slug = (str(slug).strip() if slug is not None else "")
    if not slug:
        return {"success": False, "error": "slug is required"}
    eId = (str(eId).strip() if eId is not None else "")
    if not eId:
        return {"success": False, "error": "eId is required"}
    args = {}
    args["slug"] = slug
    args["eId"] = eId
    try:
        payload = _uk_call("judgment_get_paragraph", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"judgment_get_paragraph failed: {e}"}


def case_law_grep_judgment(slug: str = None, pattern: str = None, case_insensitive: bool = True, max_hits: int = 25) -> dict:
    """Search within one UK judgment for paragraphs matching a pattern."""
    slug = (str(slug).strip() if slug is not None else "")
    if not slug:
        return {"success": False, "error": "slug is required"}
    pattern = (str(pattern).strip() if pattern is not None else "")
    if not pattern:
        return {"success": False, "error": "pattern is required"}
    args = {}
    args["slug"] = slug
    args["pattern"] = pattern
    if case_insensitive is not None:
        args["case_insensitive"] = bool(case_insensitive)
    if max_hits is not None:
        args["max_hits"] = int(max_hits)
    try:
        payload = _uk_call("case_law_grep_judgment", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"case_law_grep_judgment failed: {e}"}


def legislation_search(query: str = None, type: str = None, year: int = None, fulltext: bool = False, limit: int = 20) -> dict:
    """Search UK Acts and Statutory Instruments via the UK Legal MCP.

    A recognised short form ("HRA", "PACE", "DPA") is expanded to the Act's full title before
    the query is sent — uk-legal-mcp.fly.dev's search only matches close to the literal title,
    so the acronym alone returns nothing. See uk_legal_query.plan_uk_legislation_query and
    FORHU/chat-wonder-v2-api#94.
    """
    from uk_legal_query import plan_uk_legislation_query, prefer_exact_ref

    query = (str(query).strip() if query is not None else "")
    if not query:
        return {"success": False, "error": "query is required"}

    plan = plan_uk_legislation_query(query)

    args = {}
    args["query"] = plan.query
    if type is not None:
        args["type"] = str(type)
    if year is not None:
        args["year"] = int(year)
    if fulltext is not None:
        args["fulltext"] = bool(fulltext)
    if limit is not None:
        args["limit"] = int(limit)
    try:
        payload = _uk_with_direct_fallback(
            "legislation_search",
            args,
            lambda d: d.search_legislation(
                plan.query, type=args.get("type"), year=args.get("year"),
                fulltext=args.get("fulltext", False), limit=args.get("limit", 20),
            ),
        )
        rows = _uk_result_rows(payload)
        if rows is not None:
            from uk_legal_mcp.scoring import fill_missing_scores

            fill_missing_scores(rows, plan.query)
            if plan.ref is not None:
                rows[:] = prefer_exact_ref(rows, plan.ref)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"legislation_search failed: {e}"}


_UK_RECOMMENDATION_ROW_KEYS = (
    "title", "name", "type", "year", "number", "citation", "neutral_citation", "court", "date", "url", "uri", "link", "slug", "score",
)


def _uk_recommendation_rows(payload: dict, limit: int = 5) -> list:
    """Identifying fields only (title, citation, url ...) of the first few search rows. The snippet/body text is left
    out on purpose: nothing here should be read as a statement of what the law says."""
    rows = _uk_result_rows(payload) or []
    return [
        {k: r[k] for k in _UK_RECOMMENDATION_ROW_KEYS if k in r and r[k] not in (None, "")}
        for r in rows[:limit]
        if isinstance(r, dict)
    ]


def get_legal_recommendation_uk(legal_issue: str = None, user_context: str = None) -> dict:
    """Starting point for a general UK legal question: the UK legislation and case law most relevant to it, in one call.

    The UK sibling of get_legal_recommendation, but deliberately different: the Philippine tool has a model write the
    answer from the titles it found. The UK persona may only state law it has fetched, so this returns a research pack
    (candidate Acts/SIs and judgments with their links) for the model to read, with no generated text about the law.
    """
    issue = (str(legal_issue).strip() if legal_issue is not None else "")
    if not issue:
        return {"success": False, "error": "legal_issue is required"}

    found = {}
    errors = {}
    for key, search_fn in (("legislation", legislation_search), ("case_law", case_law_search)):
        result = search_fn(query=issue, limit=5)
        if isinstance(result, dict) and result.get("success"):
            found[key] = _uk_recommendation_rows(result)
        else:
            found[key] = []
            errors[key] = (result or {}).get("error") or "search failed"

    if len(errors) == 2:
        return {"success": False, "error": "Both the legislation and the case law search failed", "search_errors": errors}

    pack = {
        "success": True,
        "issue": issue,
        "user_context": user_context or None,
        "legislation": found["legislation"],
        "case_law": found["case_law"],
        "how_to_use": (
            "These are search hits only, not verified statements of law. Before saying what any Act, section or judgment "
            "provides, fetch it (legislation_get_section, judgment_get_paragraph) and cite it through citations_resolve. "
            "Use user_context to decide which hits matter. If neither list has an on-point hit, say you could not find one."
        ),
        "disclaimer": "This is general legal information, not legal advice. For your specific situation, please consult a qualified solicitor or barrister.",
    }
    if errors:
        pack["search_errors"] = errors
    return pack


def legislation_get_toc(type: str = None, year: int = None, number: int = None, offset: int = 0, limit: int = 200) -> dict:
    """Fetch the table of contents of a UK Act or SI."""
    type = (str(type).strip() if type is not None else "")
    if not type:
        return {"success": False, "error": "type is required"}
    if year is None:
        return {"success": False, "error": "year is required"}
    if number is None:
        return {"success": False, "error": "number is required"}
    args = {}
    args["type"] = type
    args["year"] = int(year)
    args["number"] = int(number)
    if offset is not None:
        args["offset"] = int(offset)
    if limit is not None:
        args["limit"] = int(limit)
    try:
        payload = _uk_call("legislation_get_toc", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"legislation_get_toc failed: {e}"}


def legislation_get_section(type: str = None, year: int = None, number: int = None, section: str = None, max_chars: int = 10000) -> dict:
    """Fetch the parsed text of one section of a UK Act or SI."""
    type = (str(type).strip() if type is not None else "")
    if not type:
        return {"success": False, "error": "type is required"}
    if year is None:
        return {"success": False, "error": "year is required"}
    if number is None:
        return {"success": False, "error": "number is required"}
    section = (str(section).strip() if section is not None else "")
    if not section:
        return {"success": False, "error": "section is required"}
    args = {}
    args["type"] = type
    args["year"] = int(year)
    args["number"] = int(number)
    args["section"] = section
    if max_chars is not None:
        args["max_chars"] = int(max_chars)
    try:
        payload = _uk_with_direct_fallback(
            "legislation_get_section",
            args,
            lambda d: d.get_section(type, year, number, section, max_chars=args.get("max_chars", 10000)),
        )
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"legislation_get_section failed: {e}"}


def citations_parse(text: str = None, disambiguate: bool = False) -> dict:
    """Extract and classify OSCOLA-style citations found in free text."""
    text = (str(text).strip() if text is not None else "")
    if not text:
        return {"success": False, "error": "text is required"}
    args = {}
    args["text"] = text
    if disambiguate is not None:
        args["disambiguate"] = bool(disambiguate)
    try:
        payload = _uk_call("citations_parse", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"citations_parse failed: {e}"}


def citations_resolve(citation: str = None) -> dict:
    """Parse and resolve a single OSCOLA citation, confirming it against The National Archives."""
    citation = (str(citation).strip() if citation is not None else "")
    if not citation:
        return {"success": False, "error": "citation is required"}
    args = {}
    args["citation"] = citation
    try:
        payload = _uk_call("citations_resolve", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"citations_resolve failed: {e}"}


def citations_network(case_uri: str = None) -> dict:
    """Fetch every citation a UK judgment makes (cases, legislation, SIs, retained EU law)."""
    case_uri = (str(case_uri).strip() if case_uri is not None else "")
    if not case_uri:
        return {"success": False, "error": "case_uri is required"}
    args = {}
    args["case_uri"] = case_uri
    try:
        payload = _uk_call("citations_network", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"citations_network failed: {e}"}


def citations_format_oscola(citation_type: str = None, confidence: float = None, court: str = None, year: int = None, number: int = None, resolved_url: str = None, report_series: str = None, volume: int = None, page: int = None, legislation_title: str = None, section: str = None, si_year: int = None, si_number: int = None, raw: str = None) -> dict:
    """Format an OSCOLA citation string from citations_resolve's output. Never pass hand-guessed fields."""
    citation_type = (str(citation_type).strip() if citation_type is not None else "")
    if not citation_type:
        return {"success": False, "error": "citation_type is required"}
    if confidence is None:
        return {"success": False, "error": "confidence is required"}
    args = {}
    args["citation_type"] = citation_type
    args["confidence"] = float(confidence)
    if court is not None:
        args["court"] = str(court)
    if year is not None:
        args["year"] = int(year)
    if number is not None:
        args["number"] = int(number)
    if resolved_url is not None:
        args["resolved_url"] = str(resolved_url)
    if report_series is not None:
        args["report_series"] = str(report_series)
    if volume is not None:
        args["volume"] = int(volume)
    if page is not None:
        args["page"] = int(page)
    if legislation_title is not None:
        args["legislation_title"] = str(legislation_title)
    if section is not None:
        args["section"] = str(section)
    if si_year is not None:
        args["si_year"] = int(si_year)
    if si_number is not None:
        args["si_number"] = int(si_number)
    if raw is not None:
        args["raw"] = str(raw)
    try:
        payload = _uk_call("citations_format_oscola", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"citations_format_oscola failed: {e}"}


def parliament_search_hansard(query: str = None, house: str = 'both', from_date: str = None, to_date: str = None, member_id: int = None, contribution_type: str = 'Spoken', text_mode: str = 'preview', limit: int = 20, offset: int = 0) -> dict:
    """Search Hansard debate contributions by phrase, date, house, or member."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return {"success": False, "error": "query is required"}
    args = {}
    args["query"] = query
    if house is not None:
        args["house"] = str(house)
    if from_date is not None:
        args["from_date"] = str(from_date)
    if to_date is not None:
        args["to_date"] = str(to_date)
    if member_id is not None:
        args["member_id"] = int(member_id)
    if contribution_type is not None:
        args["contribution_type"] = str(contribution_type)
    if text_mode is not None:
        args["text_mode"] = str(text_mode)
    if limit is not None:
        args["limit"] = int(limit)
    if offset is not None:
        args["offset"] = int(offset)
    try:
        payload = _uk_call("parliament_search_hansard", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_search_hansard failed: {e}"}


def parliament_policy_position_summary(topic: str = None, house: str = 'both', from_date: str = None, to_date: str = None, max_debates_scanned: int = 200) -> dict:
    """Aggregate Hansard debate-level signals on a topic without reading every contribution."""
    topic = (str(topic).strip() if topic is not None else "")
    if not topic:
        return {"success": False, "error": "topic is required"}
    args = {}
    args["topic"] = topic
    if house is not None:
        args["house"] = str(house)
    if from_date is not None:
        args["from_date"] = str(from_date)
    if to_date is not None:
        args["to_date"] = str(to_date)
    if max_debates_scanned is not None:
        args["max_debates_scanned"] = int(max_debates_scanned)
    try:
        payload = _uk_call("parliament_policy_position_summary", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_policy_position_summary failed: {e}"}


def parliament_find_member(name: str = None) -> dict:
    """Look up a Member of Parliament or Lord by name to get their integer member_id."""
    name = (str(name).strip() if name is not None else "")
    if not name:
        return {"success": False, "error": "name is required"}
    args = {}
    args["name"] = name
    try:
        payload = _uk_call("parliament_find_member", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_find_member failed: {e}"}


def parliament_member_debates(member_id: int = None, topic: str = None, limit: int = 20, offset: int = 0) -> dict:
    """Find contributions where a specific member used a topic phrase verbatim."""
    if member_id is None:
        return {"success": False, "error": "member_id is required"}
    args = {}
    args["member_id"] = int(member_id)
    if topic is not None:
        args["topic"] = str(topic)
    if limit is not None:
        args["limit"] = int(limit)
    if offset is not None:
        args["offset"] = int(offset)
    try:
        payload = _uk_call("parliament_member_debates", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_member_debates failed: {e}"}


def parliament_member_interests(member_id: int = None, category: str = None, limit: int = 20, max_description_chars: int = 500, offset: int = 0) -> dict:
    """Fetch a member's registered financial interests."""
    if member_id is None:
        return {"success": False, "error": "member_id is required"}
    args = {}
    args["member_id"] = int(member_id)
    if category is not None:
        args["category"] = str(category)
    if limit is not None:
        args["limit"] = int(limit)
    if max_description_chars is not None:
        args["max_description_chars"] = int(max_description_chars)
    if offset is not None:
        args["offset"] = int(offset)
    try:
        payload = _uk_call("parliament_member_interests", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_member_interests failed: {e}"}


def parliament_search_petitions(query: str = None, state: str = 'all', limit: int = 20, offset: int = 0) -> dict:
    """Search UK Parliament petitions by keyword."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return {"success": False, "error": "query is required"}
    args = {}
    args["query"] = query
    if state is not None:
        args["state"] = str(state)
    if limit is not None:
        args["limit"] = int(limit)
    if offset is not None:
        args["offset"] = int(offset)
    try:
        payload = _uk_call("parliament_search_petitions", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_search_petitions failed: {e}"}


def parliament_get_debate_divisions(debate_ext_id: str = None) -> dict:
    """Fetch the formal votes (divisions) held within a debate."""
    debate_ext_id = (str(debate_ext_id).strip() if debate_ext_id is not None else "")
    if not debate_ext_id:
        return {"success": False, "error": "debate_ext_id is required"}
    args = {}
    args["debate_ext_id"] = debate_ext_id
    try:
        payload = _uk_call("parliament_get_debate_divisions", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_get_debate_divisions failed: {e}"}


def parliament_get_debate_contributions(debate_ext_id: str = None, member_id: int = None) -> dict:
    """Fetch verbatim contributions from a debate, optionally filtered to one member."""
    debate_ext_id = (str(debate_ext_id).strip() if debate_ext_id is not None else "")
    if not debate_ext_id:
        return {"success": False, "error": "debate_ext_id is required"}
    args = {}
    args["debate_ext_id"] = debate_ext_id
    if member_id is not None:
        args["member_id"] = int(member_id)
    try:
        payload = _uk_call("parliament_get_debate_contributions", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_get_debate_contributions failed: {e}"}


def parliament_lookup_by_column(column_number: str = None, volume_number: int = None, house: str = 'both') -> dict:
    """Resolve an OSCOLA-style Hansard column citation to its debate."""
    column_number = (str(column_number).strip() if column_number is not None else "")
    if not column_number:
        return {"success": False, "error": "column_number is required"}
    if volume_number is None:
        return {"success": False, "error": "volume_number is required"}
    args = {}
    args["column_number"] = column_number
    args["volume_number"] = int(volume_number)
    if house is not None:
        args["house"] = str(house)
    try:
        payload = _uk_call("parliament_lookup_by_column", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"parliament_lookup_by_column failed: {e}"}


def bills_search_bills(query: str = None, house: str = None, session: int = None, stage: str = None, limit: int = 20, offset: int = 0) -> dict:
    """Search UK parliamentary bills by keyword, session, house, or stage."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return {"success": False, "error": "query is required"}
    args = {}
    args["query"] = query
    if house is not None:
        args["house"] = str(house)
    if session is not None:
        args["session"] = int(session)
    if stage is not None:
        args["stage"] = str(stage)
    if limit is not None:
        args["limit"] = int(limit)
    if offset is not None:
        args["offset"] = int(offset)
    try:
        payload = _uk_call("bills_search_bills", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"bills_search_bills failed: {e}"}


def bills_get_bill(bill_id: int = None, max_summary_chars: int = 5000) -> dict:
    """Fetch full detail for a UK parliamentary bill by bill_id."""
    if bill_id is None:
        return {"success": False, "error": "bill_id is required"}
    args = {}
    args["bill_id"] = int(bill_id)
    if max_summary_chars is not None:
        args["max_summary_chars"] = int(max_summary_chars)
    try:
        payload = _uk_call("bills_get_bill", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"bills_get_bill failed: {e}"}


def votes_search_divisions(query: str = None, house: str = 'Commons', from_date: str = None, to_date: str = None, member_id: int = None, limit: int = 25, offset: int = 0) -> dict:
    """Search Commons or Lords formal votes by topic, date, or member."""
    args = {}
    if query is not None:
        args["query"] = str(query)
    if house is not None:
        args["house"] = str(house)
    if from_date is not None:
        args["from_date"] = str(from_date)
    if to_date is not None:
        args["to_date"] = str(to_date)
    if member_id is not None:
        args["member_id"] = int(member_id)
    if limit is not None:
        args["limit"] = int(limit)
    if offset is not None:
        args["offset"] = int(offset)
    try:
        payload = _uk_call("votes_search_divisions", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"votes_search_divisions failed: {e}"}


def votes_get_division(division_id: int = None, house: str = 'Commons') -> dict:
    """Fetch the full member-by-member voting record for a division."""
    if division_id is None:
        return {"success": False, "error": "division_id is required"}
    args = {}
    args["division_id"] = int(division_id)
    if house is not None:
        args["house"] = str(house)
    try:
        payload = _uk_call("votes_get_division", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"votes_get_division failed: {e}"}


def committees_search_committees(query: str = None, house: str = None, active_only: bool = True, limit: int = 100) -> dict:
    """Search or list UK parliamentary select committees."""
    args = {}
    if query is not None:
        args["query"] = str(query)
    if house is not None:
        args["house"] = str(house)
    if active_only is not None:
        args["active_only"] = bool(active_only)
    if limit is not None:
        args["limit"] = int(limit)
    try:
        payload = _uk_call("committees_search_committees", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"committees_search_committees failed: {e}"}


def committees_get_committee(committee_id: int = None) -> dict:
    """Fetch a select committee's metadata and current membership."""
    if committee_id is None:
        return {"success": False, "error": "committee_id is required"}
    args = {}
    args["committee_id"] = int(committee_id)
    try:
        payload = _uk_call("committees_get_committee", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"committees_get_committee failed: {e}"}


def committees_search_evidence(committee_id: int = None, evidence_type: str = 'both', limit: int = 20, max_title_chars: int = 300, offset: int = 0) -> dict:
    """Fetch oral and written evidence submitted to a select committee."""
    if committee_id is None:
        return {"success": False, "error": "committee_id is required"}
    args = {}
    args["committee_id"] = int(committee_id)
    if evidence_type is not None:
        args["evidence_type"] = str(evidence_type)
    if limit is not None:
        args["limit"] = int(limit)
    if max_title_chars is not None:
        args["max_title_chars"] = int(max_title_chars)
    if offset is not None:
        args["offset"] = int(offset)
    try:
        payload = _uk_call("committees_search_evidence", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"committees_search_evidence failed: {e}"}


def hmrc_get_vat_rate(commodity_code: str = None) -> dict:
    """Look up the UK VAT rate category for a commodity or service description."""
    commodity_code = (str(commodity_code).strip() if commodity_code is not None else "")
    if not commodity_code:
        return {"success": False, "error": "commodity_code is required"}
    args = {}
    args["commodity_code"] = commodity_code
    try:
        payload = _uk_call("hmrc_get_vat_rate", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"hmrc_get_vat_rate failed: {e}"}


def hmrc_check_mtd_status(vrn: str = None) -> dict:
    """Check a UK business's Making Tax Digital VAT mandate status by VRN."""
    vrn = (str(vrn).strip() if vrn is not None else "")
    if not vrn:
        return {"success": False, "error": "vrn is required"}
    args = {}
    args["vrn"] = vrn
    try:
        payload = _uk_call("hmrc_check_mtd_status", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"hmrc_check_mtd_status failed: {e}"}


def hmrc_search_guidance(query: str = None, limit: int = 10) -> dict:
    """Search GOV.UK for official HMRC tax guidance by topic."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return {"success": False, "error": "query is required"}
    args = {}
    args["query"] = query
    if limit is not None:
        args["limit"] = int(limit)
    try:
        payload = _uk_call("hmrc_search_guidance", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"hmrc_search_guidance failed: {e}"}


def uk_legal_mcp_list_prompts() -> dict:
    """List the UK Legal MCP's available named prompts."""
    args = {}
    try:
        payload = _uk_call("list_prompts", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"uk_legal_mcp_list_prompts failed: {e}"}


def uk_legal_mcp_get_prompt(name: str = None, arguments: dict = None) -> dict:
    """Fetch one of the UK Legal MCP's named prompts by name."""
    name = (str(name).strip() if name is not None else "")
    if not name:
        return {"success": False, "error": "name is required"}
    args = {}
    args["name"] = name
    if arguments is not None:
        args["arguments"] = arguments
    try:
        payload = _uk_call("get_prompt", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"uk_legal_mcp_get_prompt failed: {e}"}


def uk_legal_mcp_list_resources() -> dict:
    """List the UK Legal MCP's available resources and resource templates."""
    args = {}
    try:
        payload = _uk_call("list_resources", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"uk_legal_mcp_list_resources failed: {e}"}


def uk_legal_mcp_read_resource(uri: str = None) -> dict:
    """Read one UK Legal MCP resource by its exact URI."""
    uri = (str(uri).strip() if uri is not None else "")
    if not uri:
        return {"success": False, "error": "uri is required"}
    args = {}
    args["uri"] = uri
    try:
        payload = _uk_call("read_resource", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"uk_legal_mcp_read_resource failed: {e}"}


# ---------------------------------------------------------------------------
# Korean Law MCP (mcp.gomdori.app/law) — [legal ai kr] persona.
# ---------------------------------------------------------------------------

def _kr_call(tool_name: str, arguments: dict) -> dict:
    """Call one Korean Law MCP tool and unwrap its JSON-RPC result."""
    from korean_law_mcp.client import get_client

    t0 = time.perf_counter()
    payload = get_client().call_tool(tool_name, arguments)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    _logger.info("korean_law_mcp %s elapsed=%.0fms", tool_name, elapsed_ms)
    return payload if isinstance(payload, dict) else {"data": payload}


def search_law(query: str = None, display: int = None) -> dict:
    """Search Korean statutes, local ordinances, and administrative rules by name keyword to get a lawId/mst."""
    query = (str(query).strip() if query is not None else "")
    if not query:
        return {"success": False, "error": "query is required"}
    args = {"query": query}
    if display is not None:
        args["display"] = int(display)
    try:
        payload = _kr_call("search_law", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"search_law failed: {e}"}


def get_law_text(mst: str = None, lawId: str = None, jo: str = None, efYd: str = None) -> dict:
    """Fetch full article text for a Korean law by mst or lawId. jo narrows to one article (e.g. '제38조'); jo='부칙' returns supplementary provisions. efYd resolves the version in force on that date."""
    mst = (str(mst).strip() if mst is not None else "")
    lawId = (str(lawId).strip() if lawId is not None else "")
    if not mst and not lawId:
        return {"success": False, "error": "mst or lawId is required"}
    args = {}
    if mst:
        args["mst"] = mst
    if lawId:
        args["lawId"] = lawId
    if jo is not None:
        args["jo"] = str(jo)
    if efYd is not None:
        args["efYd"] = str(efYd)
    try:
        payload = _kr_call("get_law_text", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"get_law_text failed: {e}"}


def get_annexes(lawName: str = None, knd: str = None, bylSeq: str = None, annexNo: str = None, query: str = None, jo: str = None, date: str = None) -> dict:
    """Fetch an annex or form (별표/서식) for a Korean law, parsed from HWP/HWPX/PDF/XLSX/DOCX."""
    lawName = (str(lawName).strip() if lawName is not None else "")
    if not lawName:
        return {"success": False, "error": "lawName is required"}
    args = {"lawName": lawName}
    if knd is not None:
        args["knd"] = str(knd)
    if bylSeq is not None:
        args["bylSeq"] = str(bylSeq)
    if annexNo is not None:
        args["annexNo"] = str(annexNo)
    if query is not None:
        args["query"] = str(query)
    if jo is not None:
        args["jo"] = str(jo)
    if date is not None:
        args["date"] = str(date)
    try:
        payload = _kr_call("get_annexes", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"get_annexes failed: {e}"}


def ordinance_radar(ordinSeq: str = None, id: str = None, ordinanceName: str = None, query: str = None) -> dict:
    """Check whether a local Korean ordinance's cited parent statute/decree has since been amended, flagging it for review. Identify the ordinance via ordinSeq/id or ordinanceName/query."""
    args = {}
    if ordinSeq is not None:
        args["ordinSeq"] = str(ordinSeq)
    if id is not None:
        args["id"] = str(id)
    if ordinanceName is not None:
        args["ordinanceName"] = str(ordinanceName)
    if query is not None:
        args["query"] = str(query)
    if not args:
        return {"success": False, "error": "ordinSeq, id, ordinanceName, or query is required"}
    try:
        payload = _kr_call("ordinance_radar", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"ordinance_radar failed: {e}"}


def legal_research(
    query: str = None,
    task: str = None,
    scenario: str = None,
    domain: str = None,
    articles: list = None,
    parentLaw: str = None,
    mst: str = None,
    lawId: str = None,
    fromDate: str = None,
    toDate: str = None,
    includeHistory: bool = None,
    text: str = None,
    maxClauses: int = None,
) -> dict:
    """Multi-step Korean legal research combining several of the MCP's APIs for one complex question.

    task selects the research shape: full_research (default, natural-language fallback),
    law_system, action_basis, dispute_prep, amendment_track, ordinance_compare,
    procedure_detail, or document_review (requires text instead of query).
    """
    args = {}
    if query is not None:
        args["query"] = str(query)
    if task is not None:
        args["task"] = str(task)
    if scenario is not None:
        args["scenario"] = str(scenario)
    if domain is not None:
        args["domain"] = str(domain)
    if articles is not None:
        args["articles"] = articles
    if parentLaw is not None:
        args["parentLaw"] = str(parentLaw)
    if mst is not None:
        args["mst"] = str(mst)
    if lawId is not None:
        args["lawId"] = str(lawId)
    if fromDate is not None:
        args["fromDate"] = str(fromDate)
    if toDate is not None:
        args["toDate"] = str(toDate)
    if includeHistory is not None:
        args["includeHistory"] = bool(includeHistory)
    if text is not None:
        args["text"] = text
    if maxClauses is not None:
        args["maxClauses"] = int(maxClauses)
    if (task or "full_research") == "document_review" and not text:
        return {"success": False, "error": "text is required when task='document_review'"}
    if (task or "full_research") != "document_review" and not query:
        return {"success": False, "error": "query is required unless task='document_review'"}
    try:
        payload = _kr_call("legal_research", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"legal_research failed: {e}"}


def legal_analysis(
    mode: str = None,
    text: str = None,
    caseNumber: str = None,
    lawName: str = None,
    jo: str = None,
    date: str = None,
    maxCitations: int = None,
    display: int = None,
    deepScan: bool = None,
    includeOrdinances: bool = None,
    includeMermaid: bool = None,
    amendments: list = None,
    title: str = None,
    outputPath: str = None,
) -> dict:
    """Verify, analyze, or diff Korean legal citations/provisions.

    mode='verify_citations' (text required) cross-checks statute/case citations in a block of text
    against the official DB to catch hallucinated citations. mode='cite_check' (caseNumber required)
    traces whether a judgment has since been overturned/superseded. mode='applicable_law'
    (lawName+date required) resolves the version of a law in force on a given date.
    mode='impact_map' (lawName+jo required) builds a reverse citation graph for one article.
    mode='amendment_table' (lawName+amendments required) builds a statutory amendment comparison table.
    """
    mode = (str(mode).strip() if mode is not None else "")
    if not mode:
        return {"success": False, "error": "mode is required"}
    args = {"mode": mode}
    if text is not None:
        args["text"] = text
    if caseNumber is not None:
        args["caseNumber"] = str(caseNumber)
    if lawName is not None:
        args["lawName"] = str(lawName)
    if jo is not None:
        args["jo"] = str(jo)
    if date is not None:
        args["date"] = str(date)
    if maxCitations is not None:
        args["maxCitations"] = int(maxCitations)
    if display is not None:
        args["display"] = int(display)
    if deepScan is not None:
        args["deepScan"] = bool(deepScan)
    if includeOrdinances is not None:
        args["includeOrdinances"] = bool(includeOrdinances)
    if includeMermaid is not None:
        args["includeMermaid"] = bool(includeMermaid)
    if amendments is not None:
        args["amendments"] = amendments
    if title is not None:
        args["title"] = str(title)
    if outputPath is not None:
        args["outputPath"] = str(outputPath)
    try:
        payload = _kr_call("legal_analysis", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"legal_analysis failed: {e}"}


def search_decisions(domain: str = None, query: str = None, display: int = None, page: int = None, sort: str = None, options: dict = None) -> dict:
    """Search Korean court decisions / administrative interpretations / tribunal rulings across 23 domains (e.g. 'precedent', 'constitutional', 'tax_tribunal', 'ftc')."""
    domain = (str(domain).strip() if domain is not None else "")
    if not domain:
        return {"success": False, "error": "domain is required"}
    args = {"domain": domain}
    if query is not None:
        args["query"] = str(query)
    if display is not None:
        args["display"] = int(display)
    if page is not None:
        args["page"] = int(page)
    if sort is not None:
        args["sort"] = str(sort)
    if options is not None:
        args["options"] = options
    try:
        payload = _kr_call("search_decisions", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"search_decisions failed: {e}"}


def get_decision_text(domain: str = None, id: str = None, full: bool = None, options: dict = None) -> dict:
    """Fetch the full text of one decision/interpretation/ruling from search_decisions by domain + id."""
    domain = (str(domain).strip() if domain is not None else "")
    id = (str(id).strip() if id is not None else "")
    if not domain or not id:
        return {"success": False, "error": "domain and id are required"}
    args = {"domain": domain, "id": id}
    if full is not None:
        args["full"] = bool(full)
    if options is not None:
        args["options"] = options
    try:
        payload = _kr_call("get_decision_text", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"get_decision_text failed: {e}"}


def discover_tools_kr(intent: str = None) -> dict:
    """Discover the Korean Law MCP's 80+ specialized tools (tax tribunal, customs, constitutional court, treaties, etc.) by intent/category keyword."""
    intent = (str(intent).strip() if intent is not None else "")
    if not intent:
        return {"success": False, "error": "intent is required"}
    args = {"intent": intent}
    try:
        payload = _kr_call("discover_tools", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"discover_tools_kr failed: {e}"}


def execute_tool_kr(tool_name: str = None, params: dict = None) -> dict:
    """Run one of the Korean Law MCP's specialized tools (found via discover_tools_kr) by name and arguments."""
    tool_name = (str(tool_name).strip() if tool_name is not None else "")
    if not tool_name:
        return {"success": False, "error": "tool_name is required"}
    args = {"tool_name": tool_name, "params": params or {}}
    try:
        payload = _kr_call("execute_tool", args)
        return {"success": True, **payload}
    except Exception as e:
        return {"success": False, "error": str(e), "message": f"execute_tool_kr failed: {e}"}


# ---------------------------------------------------------------------------
# UK (England & Wales) document drafting — the UK sibling of generate_legal_document,
# which is Philippine-only. Layouts follow the Practice Direction on Pre-Action Conduct (para 6),
# CPR PD 32 (paras 17-20, witness statements), the Statutory Declarations Act 1835 and the Powers
# of Attorney Act 1971 s.10. England & Wales only — Scotland and Northern Ireland use different
# court forms and terminology.
# ---------------------------------------------------------------------------

UK_DRAFTER_SYSTEM_PROMPT = (
    "You are a drafter of legal documents for England and Wales. Write in British English, in formal "
    "but plain modern legal English. Produce the complete, final document text — never a description, "
    "outline or explanation of what the document should contain. Use ONLY the facts supplied; never "
    "invent names, dates, amounts, addresses, claim numbers or exhibit contents — where a detail is "
    "missing, leave a blank written [___]. Cite only the rules and statutory provisions named in the "
    "instructions; never cite case law and never invent a citation. Output only the document itself — "
    "no commentary before or after it, and no markdown code fences."
)

# Fixed-form documents (letters, declarations, deeds — generate_legal_document_uk) are formatted with a
# light markdown skin that the docx/PDF exporter understands (see generated-document-renderer.ts): a
# single-line heading for the title and **bold** for labels and signature names. Court pleadings
# (draft_pleading_uk) keep the plain CPR-style caption instead, so this note is not added there.
UK_FIXED_FORM_SYSTEM_PROMPT = UK_DRAFTER_SYSTEM_PROMPT + (
    " Format the document in Markdown: give it one single-line heading (any level from # to ######, alone "
    "on its own line with a blank line before and after) for its title, and use **bold** for field labels "
    "(e.g. **Date:**, **To:**, **Private and Confidential**) and for names in signature blocks. Do not use "
    "bullet or numbered markdown lists, blockquotes, inline code or code fences — write numbered paragraphs "
    "as plain text, e.g. '1. ...', not as a markdown list."
)

UK_DISCLAIMER = (
    "This is an AI-drafted document for England and Wales, prepared from public rules and forms. It has not "
    "been reviewed by a solicitor. Check it against the current rules and get it reviewed before you sign, "
    "serve or file it."
)

UK_DEFAULT_NEXT_STEPS = [
    "Check every detail for accuracy",
    "Fill in any blanks marked [___]",
    "Get the document reviewed by a solicitor before relying on it",
]

UK_DOCUMENT_TEMPLATES = {
    "witness_statement": {
        "name": "Witness Statement",
        "required_fields": ["witness_name", "witness_address", "party_role", "statement_facts"],
        "optional_fields": ["witness_occupation", "court", "claim_number", "case_title", "statement_number", "exhibits"],
        "next_steps": [
            "Check every fact is within your own knowledge, or say where the information came from",
            "Mark and refer to each exhibit exactly as listed, and attach the exhibits",
            "Sign and date it yourself — the statement of truth is yours",
            "If the case is in the Business and Property Courts, extra requirements apply to trial witness statements — check the current rules",
        ],
        "prompt": """Draft a witness statement for civil proceedings in England and Wales, laid out as Practice Direction 32 (paras 17-20) requires.

Facts supplied:
- Witness: {witness_name}
- Address (or work address if giving evidence in a professional capacity): {witness_address}
- Occupation / position: {witness_occupation}
- Party on whose behalf it is made: {party_role}
- Court: {court}
- Claim number: {claim_number}
- Title of proceedings: {case_title}
- Statement number: {statement_number}
- Exhibits: {exhibits}
- Evidence the witness gives: {statement_facts}

Layout:
1. Top right corner (PD 32 para 17.2): party on whose behalf the statement is made; the witness's initials and surname; the number of the statement (e.g. 1st); identifying initials and numbers of exhibits; the date the statement was made.
2. Court name, claim number and title of the proceedings.
3. Title: a single-line heading "### Witness Statement of [Full Name In Capitals]", alone on its own line.
4. Opening in the first person (para 18.1): full name, address (or work address), occupation or description, whether a party or an employee of a party, then "state as follows:".
5. The evidence in consecutively numbered paragraphs, in the first person, each dealing with a distinct point. Say which matters are within the witness's own knowledge and which are information and belief, giving the source of that information. Put all dates and numbers in figures. Refer to each exhibit by its identifying mark.
6. Statement of truth, in exactly these words: "I believe that the facts stated in this witness statement are true. I understand that proceedings for contempt of court may be brought against anyone who makes, or causes to be made, a false statement in a document verified by a statement of truth without an honest belief in its truth."
7. A signature line and a date line, with the witness's printed name in **bold**.""",
    },
    "statutory_declaration": {
        "name": "Statutory Declaration",
        "required_fields": ["declarant_name", "declarant_address", "declaration_facts"],
        "optional_fields": ["declarant_occupation", "declaration_purpose"],
        "next_steps": [
            "Do not sign it in advance — you must sign it in front of a solicitor, commissioner for oaths or notary public",
            "Take photo ID and proof of address; the person taking it may charge a fee",
            "Check whether the organisation asking for it has its own required wording",
        ],
        "prompt": """Draft a statutory declaration under the Statutory Declarations Act 1835 (England and Wales).

Facts supplied:
- Declarant: {declarant_name}
- Address: {declarant_address}
- Occupation: {declarant_occupation}
- Purpose: {declaration_purpose}
- What is being declared: {declaration_facts}

Layout:
1. Title: a single-line heading "### Statutory Declaration", alone on its own line.
2. Opening: "I, [full name], of [address], [occupation], do solemnly and sincerely declare as follows:"
3. The facts in consecutively numbered paragraphs, in the first person, stating only what the declarant says is true. If the declaration is about a lost item, cover what it is, when and where it was last seen, what was done to look for it, and that it has not been found.
4. Closing, in exactly these words: "AND I make this solemn declaration conscientiously believing the same to be true, and by virtue of the provisions of the Statutory Declarations Act 1835."
5. A signature line for the declarant in **bold**, then: "Declared at [place] this [day] of [month] [year]" and "Before me," followed by a signature line and blanks for the name and capacity of the person taking the declaration (solicitor, commissioner for oaths or notary public).""",
    },
    "letter_before_claim": {
        "name": "Letter Before Claim",
        "required_fields": ["sender_name", "recipient_name", "claim_summary", "remedy_sought"],
        "optional_fields": [
            "sender_address", "recipient_address", "reference_number", "amount", "how_calculated",
            "documents_relied_on", "response_days",
        ],
        "next_steps": [
            "If you are claiming a debt from an individual (including a sole trader), the Pre-Action Protocol for Debt Claims is likely to apply instead — it has its own required contents, information sheet and reply form",
            "Keep proof of when and how you sent the letter",
            "Keep copies of the documents you refer to",
        ],
        "prompt": """Draft an open letter before claim for England and Wales, following paragraph 6 of the Practice Direction on Pre-Action Conduct.

Facts supplied:
- From: {sender_name}, {sender_address}
- To: {recipient_name}, {recipient_address}
- Reference: {reference_number}
- Summary of the claim and its basis: {claim_summary}
- What the sender wants: {remedy_sought}
- Amount claimed: {amount}
- How the amount is calculated: {how_calculated}
- Documents relied on: {documents_relied_on}
- Days allowed to respond: {response_days}

The letter must:
1. A single-line heading "### Letter Before Claim", alone on its own line.
2. Be dated, with both addresses and the reference — put the labels **Date:**, **From:**, **To:** and **Reference:** in bold where a value is given for them.
3. Give concise details of the claim: the basis on which it is made and a summary of the facts.
4. State clearly what the sender wants the recipient to do, and, where money is claimed, how the amount is calculated.
5. List the key documents the claim relies on and say copies are enclosed or available.
6. Allow a reasonable time to respond — use the number of days given above, or 14 days if it says [___].
7. Ask the recipient to reply saying whether the claim is accepted and, if not, why, which facts and parts of the claim are disputed, and whether they intend to make a counterclaim.
8. Say the sender is willing to consider alternative dispute resolution, such as mediation, if the recipient prefers it.
9. Say that if there is no satisfactory response the sender may start court proceedings without further notice, will ask the court to take the recipient's non-compliance with the pre-action requirements into account on costs, and will claim interest and costs.
10. Close formally, with a signature block for the sender, their printed name in **bold**.""",
    },
    "general_power_of_attorney": {
        "name": "General Power of Attorney",
        "required_fields": ["donor_name", "attorney_name"],
        "optional_fields": ["donor_address", "attorney_address", "appointment_type", "limits_on_authority"],
        "next_steps": [
            "It must be signed as a deed, in front of an independent adult witness who also signs",
            "This ends if you lose mental capacity — for that you need a lasting power of attorney, which uses official forms from the Office of the Public Guardian",
            "Banks and other organisations may ask to see the original or a certified copy",
        ],
        "prompt": """Draft a general power of attorney under section 10 of the Powers of Attorney Act 1971 (England and Wales), executed as a deed.

Facts supplied:
- Donor (person giving the power): {donor_name}, {donor_address}
- Attorney(s): {attorney_name}, {attorney_address}
- If more than one attorney — jointly, or jointly and severally: {appointment_type}
- Limits on the authority given: {limits_on_authority}

Layout:
1. Title: a single-line heading "### General Power of Attorney", alone on its own line.
2. Operative wording: "THIS GENERAL POWER OF ATTORNEY is made this [day] of [month] [year] by [donor] of [address]. I appoint [attorney] of [address] [jointly / jointly and severally, only if there is more than one attorney] to be my attorney[s] in accordance with section 10 of the Powers of Attorney Act 1971."
3. If limits on authority were given, add a clause setting them out precisely. If none were given, do NOT add any limit.
4. Execution as a deed: "Executed as a deed by [donor] in the presence of:" with the donor's printed name in **bold** on the signature line, and a witness block (signature, full name, address, occupation).""",
    },
    "letter_of_authority": {
        "name": "Letter of Authority",
        "required_fields": ["authoriser_name", "authorised_person", "authority_purpose"],
        "optional_fields": ["authoriser_address", "organisation_addressed", "reference_number", "valid_until"],
        "next_steps": [
            "Sign and date it, and enclose ID if the organisation asks for it",
            "Check whether the organisation requires its own authority form instead",
        ],
        "prompt": """Draft a letter of authority (a letter authorising another person to act on the writer's behalf) for use in England and Wales.

Facts supplied:
- Person giving authority: {authoriser_name}, {authoriser_address}
- Person authorised: {authorised_person}
- Organisation the letter is addressed to: {organisation_addressed}
- Reference: {reference_number}
- What the person is authorised to do: {authority_purpose}
- Valid until: {valid_until}

Include: a single-line heading "### Letter of Authority" alone on its own line; date, addressee and reference, with the labels **Date:**, **To:** and **Reference:** in bold where a value is given for them; a clear statement that the writer authorises the named person to act for them for the stated purpose only; the period of validity (or that it lasts until withdrawn in writing); a line that the organisation may deal with the named person and discuss the matter with them; the writer's signature line and printed name in **bold**.""",
    },
    "loan_agreement": {
        "name": "Loan Agreement",
        "required_fields": ["lender_name", "borrower_name", "principal_amount", "repayment_terms"],
        "optional_fields": ["lender_address", "borrower_address", "interest_rate", "loan_date", "security"],
        "next_steps": [
            "Both parties should sign and date it and each keep a copy",
            "If you lend money as a business, or the borrower is a consumer, consumer credit rules may apply — get advice first",
            "Keep records of every payment made",
        ],
        "prompt": """Draft a simple loan agreement governed by the law of England and Wales.

Facts supplied:
- Lender: {lender_name}, {lender_address}
- Borrower: {borrower_name}, {borrower_address}
- Amount lent: {principal_amount}
- Date of loan: {loan_date}
- Interest: {interest_rate}
- Repayment terms: {repayment_terms}
- Security: {security}

Start with a single-line heading "### Loan Agreement" alone on its own line. Include clauses for: parties (put the lender's and borrower's names in **bold** where first introduced); the loan and the date the money is advanced; interest (say the loan is interest-free if no rate is given); repayment; what happens on late payment or default; that the borrower may repay early without penalty unless the terms above say otherwise; no assignment without written consent; that the agreement is the whole agreement and can be varied only in writing; governing law and the courts of England and Wales. End with signature blocks for both parties, each with a date line and the signatory's printed name in **bold**.""",
    },
    "deed_poll": {
        "name": "Deed Poll (Change of Name)",
        "required_fields": ["former_name", "new_name", "address"],
        "optional_fields": ["date_of_birth"],
        "next_steps": [
            "Sign it as a deed in front of an independent adult witness (some organisations ask for two)",
            "An unenrolled deed poll is available from age 16; enrolling it on the public record through the High Court is available from age 18 — check the current fee on GOV.UK",
            "Ask your bank, passport office and other organisations which type they accept",
        ],
        "prompt": """Draft a deed poll for a change of name in England and Wales.

Facts supplied:
- Former name: {former_name}
- New name: {new_name}
- Address: {address}
- Date of birth: {date_of_birth}

Layout:
1. Title: a single-line heading "### Deed Poll", alone on its own line.
2. Opening: "I, [NEW NAME], of [address], [born on date of birth, if given,] formerly known as [FORMER NAME], hereby:"
3. Numbered declarations: (1) renounce, relinquish and abandon the use of my former name and declare that I have assumed, adopted and determined to take and use from the date of this deed the name [new name] in substitution for it; (2) declare that I shall at all times hereafter in all records, deeds, documents and other writings and in all actions and proceedings, as well as in all dealings and transactions and on all occasions whatsoever, use and subscribe the name [new name] as my name in place of the former name; (3) authorise and require all persons at all times to designate, address and describe me by the name [new name].
4. "In witness whereof I have signed this deed on [day] of [month] [year]," then "Signed as a deed by the above-named in the presence of:" with the person's printed name in **bold** on the signature line, and a witness block (signature, full name, address, occupation).""",
    },
    "employment_termination_notice": {
        "name": "Notice of Termination of Employment",
        "required_fields": [
            "company_name", "employee_name", "job_title", "termination_reason", "notice_period", "termination_date",
        ],
        "optional_fields": [
            "company_address", "employee_address", "notice_start_date", "during_notice",
            "appeal_contact_name", "appeal_days", "manager_name", "manager_job_title", "letter_date",
        ],
        "next_steps": [
            "Check the notice period meets or exceeds the statutory minimum under section 86 of the Employment Rights Act 1996, based on the employee's length of service",
            "Follow a fair dismissal process (investigation, meeting, right to be accompanied) before sending this, particularly for conduct or capability reasons — sending notice without one raises unfair dismissal risk",
            "Confirm the employment contract actually allows payment in lieu of notice before using it",
            "Get the letter reviewed by a solicitor or HR professional before it is sent",
        ],
        "prompt": """Draft a notice of termination of employment letter for England and Wales.

Facts supplied:
- Company: {company_name}, {company_address}
- Employee: {employee_name}, {employee_address}
- Job title: {job_title}
- Reason for termination: {termination_reason}
- Notice period: {notice_period}
- Notice start date: {notice_start_date}
- Termination date: {termination_date}
- During the notice period: {during_notice}
- Person to receive an appeal: {appeal_contact_name}
- Days allowed to appeal: {appeal_days}
- Signed by: {manager_name}, {manager_job_title}
- Date of letter: {letter_date}

Format the output in Markdown, following this structure exactly:
1. "**{company_name}**" alone on its own line, then the company address on the following lines, no markdown.
2. A blank line, then "**Date:** [letter date]".
3. A blank line, then "**Private and Confidential**".
4. A blank line, then "**To:** [employee name]" followed by the employee's address on the next line.
5. A blank line, then the heading "### Notice of Termination of Employment" alone on its own line, on a line by itself with nothing else.
6. A blank line, then "Dear [employee name],".
7. Body paragraphs in plain prose, no bullet points, each separated by a blank line, covering in this order:
   - A formal statement that the company is terminating the employee's employment in the stated job title.
   - The reason for termination, stated clearly from the facts given — do not embellish or add detail not supplied.
   - The notice period, referring to the employment contract, stating the notice start date and the termination date. Note that the notice given must be at least the statutory minimum under section 86 of the Employment Rights Act 1996; do not state a specific number of weeks under that section unless the employee's length of service was given in the facts — otherwise refer to the statutory minimum in general terms.
   - What happens during the notice period, based on the facts given (continuing to work normally, garden leave, or payment in lieu of notice under the contract).
   - Final pay: salary and other contractual payments due up to the termination date, plus accrued but unused holiday pay, subject to lawful deductions.
   - A reminder of continuing obligations under the employment contract that survive termination (confidentiality, intellectual property, data protection, post-termination restrictions).
   - The right to appeal: if a contact and number of days were given, state them; otherwise state that the employee has the right to appeal and should raise it promptly in writing.
   - A short closing paragraph thanking the employee for their service and wishing them well.
8. A blank line, then "Yours sincerely,".
9. A blank line, then "**[manager name]**" alone on its own line, followed by their job title and the company name on their own following lines, no markdown.

Never invent a reason, date, amount or notice period not supplied — write [___] for anything missing.""",
    },
}


def generate_legal_document_uk(document_type: str, details: dict = None, format: str = "docx", **kwargs) -> dict:
    """Generate a legal document for England and Wales from a fixed template."""
    from openai import OpenAI

    if details is None:
        details = {}
    if kwargs:
        details.update(kwargs)

    doc_type_lower = document_type.lower().replace(" ", "_").replace("-", "_")
    if doc_type_lower not in UK_DOCUMENT_TEMPLATES:
        return {
            "success": False,
            "error": f"Unknown document type: '{document_type}'",
            "available_types": list(UK_DOCUMENT_TEMPLATES.keys()),
        }

    template = UK_DOCUMENT_TEMPLATES[doc_type_lower]
    missing = [f for f in template["required_fields"] if not details.get(f)]
    if missing:
        return {
            "success": False,
            "error": "Missing required fields",
            "missing_fields": missing,
            "required_fields": template["required_fields"],
            "optional_fields": template.get("optional_fields", []),
        }

    filled = dict(details)
    for field in template.get("optional_fields", []):
        if not filled.get(field):
            filled[field] = "[___]"

    client = make_openai_client(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": UK_FIXED_FORM_SYSTEM_PROMPT},
                {"role": "user", "content": template["prompt"].format(**filled)},
            ],
            temperature=0.2,
        )
        return {
            "success": True,
            "document_type": doc_type_lower,
            "document_name": template["name"],
            "format": format,
            "content": response.choices[0].message.content,
            "fields_used": details,
            "disclaimer": UK_DISCLAIMER,
            "next_steps": template.get("next_steps", UK_DEFAULT_NEXT_STEPS),
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def draft_pleading_uk(
    pleading_type: str,
    case_facts: str,
    grounds: str,
    relief_sought: str = None,
    responding_to: str = None,
    court: str = None,
    case_title: str = None,
    claim_number: str = None,
    party_role: str = None,
    format: str = "docx",
    authorities: list = None,
    **kwargs,
) -> dict:
    """Draft a complete statement of case or similar court document for England and Wales, in full prose."""
    from openai import OpenAI

    missing = [f for f, v in (("pleading_type", pleading_type), ("case_facts", case_facts), ("grounds", grounds)) if not v]
    if missing:
        return {"success": False, "error": "Missing required fields", "missing_fields": missing}

    court = court or "[___]"
    case_title = case_title or "[___]"
    claim_number = claim_number or "[___]"
    party_role = party_role or "[___]"
    responding_to_block = f"\n- Responding to: {responding_to}" if responding_to else ""
    relief_block = (
        f"\n- Relief sought: {relief_sought}" if relief_sought
        else "\n- Relief sought: not specified — infer the standard claim or order sought for this document type from the grounds given."
    )
    authorities_block, _ = _format_authorities_block(authorities)
    if authorities:
        citation_scope = (
            "Cite the CPR rules and practice directions named above, plus the case law and legislation "
            "listed under Authorities above, exactly as given there. Do not cite any case law or statute "
            "that is not listed under Authorities — if a point needs an authority you don't have, leave it "
            "unsupported rather than naming a case or section from memory."
        )
    else:
        citation_scope = (
            "Cite only the CPR rules and practice directions named above. Do not cite case law or any statute."
        )

    prompt = f"""Draft a complete {pleading_type} for civil proceedings in England and Wales, ready to file.

- Court: {court}
- Claim number: {claim_number}
- Title of proceedings: {case_title}
- Party this document is for: {party_role}{responding_to_block}
- Facts: {case_facts}
- Grounds: {grounds}{relief_block}{authorities_block}

Write the full document now — not an outline, not a summary of what it should say. Follow this layout:
1. Heading: the name of the court (if none given, use "IN THE COUNTY COURT AT [___]" or "IN THE HIGH COURT OF JUSTICE, KING'S BENCH DIVISION" as suits the claim), the claim number, the title of the proceedings with each party and their role (Claimant / Defendant), and the title of this document.
2. The body in consecutively numbered paragraphs, each dealing with one point, applying the facts supplied to the grounds.
3. Follow the rules for the document type:
   - Particulars of Claim: a concise statement of the facts relied on (CPR 16.4(1)(a)); if interest is claimed, whether it is claimed under a contract, an enactment or another basis, with the rate, the dates and, for a specified sum, the amount and daily rate (16.4(1)(b) and (2)); a statement if aggravated, exemplary or provisional damages are claimed (16.4(1)(c) and (d)). For a contract claim, identify the contract and its relevant terms; for an oral contract, the words used, by whom, to whom, when and where (PD 16 paras 7.3-7.5). End with what the Claimant claims.
   - Defence (or Defence and Counterclaim): deal with every allegation in the Particulars of Claim, stating which are admitted, which are denied and which the Defendant cannot admit or deny but requires the Claimant to prove (CPR 16.5(1)); give reasons for every denial and, if the Defendant puts forward a different version of events, state it (16.5(2)); if the value claimed is disputed, say so with reasons and an alternative figure where possible (16.5(6)). Plead limitation if relied on.
   - Reply, or any other document responding to another: answer the points raised, paragraph by paragraph.
4. Statement of truth (CPR Part 22 and PD 22), in these words: "I believe that the facts stated in [this / these] [name of document, in lower case, e.g. these particulars of claim, this defence] are true. I understand that proceedings for contempt of court may be brought against anyone who makes, or causes to be made, a false statement in a document verified by a statement of truth without an honest belief in its truth." Follow with a signature line, printed name, position (e.g. Claimant) and date.
5. Finish with the date and the party's name and address for service (leave blanks if not supplied).

{citation_scope} If interest is claimed but the user gave no rate, start date or legal basis, write those as [___] — never state a rate or a statutory basis the user did not give. Do not add alternative causes of action or heads of loss the user did not raise. Leave a blank [___] for anything not supplied — never make up a fact, date, name or amount."""

    client = make_openai_client(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": UK_DRAFTER_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return {
            "success": True,
            "pleading_type": pleading_type,
            "format": format,
            "content": response.choices[0].message.content,
            "disclaimer": UK_DISCLAIMER,
            "next_steps": [
                "Check every fact, date and amount against your documents",
                "Fill in any blanks marked [___], including the court, claim number and your address for service",
                "If a solicitor signs the statement of truth for you, the wording changes — they should confirm it",
                "Check the court's current time limits and fees before filing",
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def get_legal_recommendation(legal_issue: str, user_context: str = None) -> dict:
    """Provide legal information and recommendations for a given legal issue."""
    from openai import OpenAI

    relevant_materials = []
    for search_fn in (search_jurisprudence, search_republic_acts):
        search_result = search_fn(legal_issue, limit=2)
        if search_result.get("success"):
            for doc in search_result.get("results", [])[:2]:
                relevant_materials.append(
                    f"{doc.get('type', 'Document')}: {doc.get('title', '')} ({doc.get('url', '')})"
                )

    client = make_openai_client(api_key=os.getenv("OPENAI_API_KEY"))
    context_section = f"\n\nUser's Situation: {user_context}" if user_context else ""
    materials_section = (
        "\n\nRelevant Legal Materials Found:\n" + "\n".join(relevant_materials)
        if relevant_materials
        else ""
    )

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Philippine legal information assistant. Provide helpful legal INFORMATION (not advice).\n"
                        "IMPORTANT RULES:\n"
                        "1. Start with: 'This is general legal information, not legal advice.'\n"
                        "2. Structure your response with clear sections\n"
                        "3. Cite relevant Philippine laws and cases when applicable\n"
                        "4. Always recommend consulting a licensed attorney for specific situations\n"
                        "5. End with: 'For your specific situation, please consult a licensed attorney.'"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Legal Issue: {legal_issue}{context_section}{materials_section}\n\n"
                        "Please provide:\n"
                        "1. BRIEF ANSWER - Direct response to the question\n"
                        "2. LEGAL BASIS - Relevant Philippine law or case\n"
                        "3. EXPLANATION - Plain language explanation\n"
                        "4. PRACTICAL STEPS - What the person can do\n"
                        "5. WHEN TO SEEK LEGAL HELP - Signs they need a lawyer"
                    ),
                },
            ],
            temperature=0.3,
        )
        return {
            "success": True,
            "issue": legal_issue,
            "recommendation": response.choices[0].message.content,
            "relevant_materials": relevant_materials,
            "disclaimer": "This is general legal information, not legal advice. For your specific situation, please consult a licensed attorney.",
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def analyze_document(s3_key: str, filename: str = None) -> dict:
    """Download a document from S3 and return a structured Philippine legal analysis."""
    import s3_storage
    from openai import OpenAI

    fname = filename or os.path.basename(s3_key)
    ext = os.path.splitext(fname)[1].lower()

    tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(s3_key))
    downloaded = s3_storage.download_from_s3(s3_key, tmp_path)
    if not downloaded:
        return {"success": False, "error": "File not found in S3 or S3 is not configured."}

    extracted_text = ""
    try:
        with open(tmp_path, "rb") as f:
            contents = f.read()

        if len(contents) > 20 * 1024 * 1024:
            return {"success": False, "error": "File too large. Maximum allowed size is 20MB."}

        if ext == ".txt":
            for enc in ["utf-8", "cp1252", "latin-1"]:
                try:
                    extracted_text = contents.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue

        elif ext == ".pdf":
            try:
                import PyPDF2
                reader = PyPDF2.PdfReader(io.BytesIO(contents))
                extracted_text = "\n\n".join(
                    p.extract_text().strip() for p in reader.pages if p.extract_text()
                )
            except ImportError:
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(contents)) as pdf:
                        extracted_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages).strip()
                except ImportError:
                    return {"success": False, "error": "PDF library not installed. Run: pip install PyPDF2"}
            except Exception as e:
                return {"success": False, "error": f"Failed to parse PDF: {e}"}

        elif ext in (".docx", ".doc"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(contents))
                extracted_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                return {"success": False, "error": "DOCX library not installed. Run: pip install python-docx"}
            except Exception as e:
                return {"success": False, "error": f"Failed to parse DOCX: {e}"}

        elif ext in (".xlsx", ".xlsm", ".xlam"):
            try:
                import openpyxl
                wb = openpyxl.load_workbook(io.BytesIO(contents), data_only=True, read_only=True)
                sheet_texts = []
                for ws in wb.worksheets:
                    rows = []
                    for row in ws.iter_rows(values_only=True):
                        if any(cell is not None for cell in row):
                            rows.append("\t".join("" if cell is None else str(cell) for cell in row))
                    if rows:
                        sheet_texts.append(f"--- Sheet: {ws.title} ---\n" + "\n".join(rows))
                extracted_text = "\n\n".join(sheet_texts)
            except ImportError:
                return {"success": False, "error": "Excel library not installed. Run: pip install openpyxl"}
            except Exception as e:
                return {"success": False, "error": f"Failed to parse Excel file: {e}"}

        elif ext in (".mp3", ".wav", ".m4a", ".mp4"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"success": False, "error": "OpenAI API key required for audio transcription."}
            try:
                client = make_openai_client(api_key=api_key)
                with open(tmp_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                extracted_text = transcription.text
            except Exception as e:
                return {"success": False, "error": f"Audio transcription failed: {e}"}

        elif ext in (".png", ".jpg", ".jpeg", ".webp", ".heic"):
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return {"success": False, "error": "OpenAI API key required for image OCR."}
            try:
                import base64, mimetypes
                if ext == ".heic":
                    try:
                        import pillow_heif
                        from PIL import Image
                        pillow_heif.register_heif_opener()
                        image = Image.open(io.BytesIO(contents)).convert("RGB")
                        buf = io.BytesIO()
                        image.save(buf, format="JPEG")
                        contents = buf.getvalue()
                        mime_type = "image/jpeg"
                    except ImportError:
                        return {"success": False, "error": "HEIC library not installed. Run: pip install pillow-heif Pillow"}
                else:
                    mime_type = mimetypes.guess_type(fname)[0] or f"image/{ext[1:]}"
                b64 = base64.b64encode(contents).decode("utf-8")
                client = make_openai_client(api_key=api_key)
                resp = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Extract all text from this image exactly as written. If no text, describe the image briefly."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ]}],
                    max_tokens=3000,
                )
                extracted_text = resp.choices[0].message.content.strip()
            except Exception as e:
                return {"success": False, "error": f"Image OCR failed: {e}"}

        else:
            return {"success": False, "error": f"Unsupported file type '{ext}'. Supported: PDF, DOCX, TXT, XLSX, XLSM, XLAM, PNG, JPG, WEBP, HEIC, MP3, WAV, M4A, MP4."}

        extracted_text = extracted_text.strip()
        if not extracted_text:
            return {"success": False, "error": "No text could be extracted from the document."}

        CHAR_LIMIT = 50000
        truncated = len(extracted_text) > CHAR_LIMIT
        if truncated:
            extracted_text = extracted_text[:CHAR_LIMIT]

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"success": False, "error": "OpenAI API key not configured."}

        client = make_openai_client(api_key=api_key)
        model = os.getenv("CHAT_MODEL", "gpt-4o-mini")

        system_prompt = (
            "You are an expert Philippine legal document analyst with deep knowledge of the Civil Code, "
            "Revised Penal Code, Labor Code, and Supreme Court decisions. "
            "Analyze the document and produce a structured legal analysis with these sections:\n\n"
            "## 1. Document Overview\n"
            "Document type, parties, date, jurisdiction, and legal purpose.\n\n"
            "## 2. Key Legal Issues & Provisions\n"
            "All significant obligations, rights, conditions, and prohibitions with their implications.\n\n"
            "## 3. Relevant Philippine Laws & Jurisprudence\n"
            "Applicable statutes (Civil Code Articles, RA numbers) and Supreme Court decisions (G.R. numbers).\n\n"
            "## 4. Notable Clauses or Concerns\n"
            "Unusual, ambiguous, or risky clauses with explanation of the legal risk.\n\n"
            "## 5. Parties' Rights & Obligations\n"
            "What each party is entitled to and obligated to do.\n\n"
            "## 6. Potential Legal Issues or Disputes\n"
            "Scenarios that could lead to disputes or enforcement problems and how to mitigate them.\n\n"
            "## 7. Recommendations\n"
            "Specific, actionable legal advice and suggested next steps."
        )
        text_for_ai = extracted_text[:25000]
        if truncated:
            text_for_ai += f"\n\n[Note: Document truncated — only the first {CHAR_LIMIT:,} characters were analyzed.]"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Document: **{fname}**\n\n---\n\n{text_for_ai}"},
            ],
            temperature=0.2,
            max_tokens=4000,
        )
        ai_summary = response.choices[0].message.content.strip()

        file_url = s3_storage.generate_presigned_get(s3_key)

        return {
            "success": True,
            "filename": fname,
            "s3_key": s3_key,
            "file_url": file_url,
            "ai_summary": ai_summary,
            "char_count": len(extracted_text),
            "truncated": truncated,
        }

    except Exception as e:
        logging.error(f"[analyze_document] Unexpected error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


_UK_ANALYSIS_CHAR_LIMIT = 50000
_UK_ANALYSIS_MAX_BYTES = 20 * 1024 * 1024

_UK_ANALYSIS_GUIDE = (
    "This is the extracted text of the user's document, not an analysis. Write the analysis yourself, for England & Wales, "
    "under these headings: 1. Document overview (type, parties, date, purpose); 2. Key terms and obligations; "
    "3. England & Wales law that may apply; 4. Unusual, ambiguous or risky clauses; 5. Each party's rights and obligations; "
    "6. Possible disputes and how to reduce them; 7. Suggested next steps. Under heading 3, name only law you have fetched "
    "with legislation_search / legislation_get_section or case_law_search / judgment_get_paragraph, and cite it through "
    "citations_resolve and citations_format_oscola. If you have not fetched it, do not state what an Act or case provides; "
    "say which area of law to check. Quote the document's own words when pointing at a clause."
)


def _uk_extract_document_text(path: str, fname: str, contents: bytes) -> dict:
    """Text of an uploaded document (txt, pdf, docx, image OCR, audio transcript). {"text": ...} or {"error": ...}."""
    ext = os.path.splitext(fname)[1].lower()
    if ext == ".txt":
        for enc in ("utf-8", "cp1252", "latin-1"):
            try:
                return {"text": contents.decode(enc)}
            except UnicodeDecodeError:
                continue
        return {"error": "Could not decode the text file."}
    if ext == ".pdf":
        try:
            import PyPDF2
            reader = PyPDF2.PdfReader(io.BytesIO(contents))
            return {"text": "\n\n".join(t.strip() for t in (p.extract_text() for p in reader.pages) if t)}
        except ImportError:
            try:
                import pdfplumber
                with pdfplumber.open(io.BytesIO(contents)) as pdf:
                    return {"text": "\n\n".join(p.extract_text() or "" for p in pdf.pages).strip()}
            except ImportError:
                return {"error": "PDF library not installed. Run: pip install PyPDF2"}
        except Exception as e:
            return {"error": f"Failed to parse PDF: {e}"}
    if ext in (".docx", ".doc"):
        try:
            import docx
            doc = docx.Document(io.BytesIO(contents))
            return {"text": "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())}
        except ImportError:
            return {"error": "DOCX library not installed. Run: pip install python-docx"}
        except Exception as e:
            return {"error": f"Failed to parse DOCX: {e}"}
    if ext in (".xlsx", ".xlsm", ".xlam"):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(io.BytesIO(contents), data_only=True, read_only=True)
            sheet_texts = []
            for ws in wb.worksheets:
                rows = []
                for row in ws.iter_rows(values_only=True):
                    if any(cell is not None for cell in row):
                        rows.append("\t".join("" if cell is None else str(cell) for cell in row))
                if rows:
                    sheet_texts.append(f"--- Sheet: {ws.title} ---\n" + "\n".join(rows))
            return {"text": "\n\n".join(sheet_texts)}
        except ImportError:
            return {"error": "Excel library not installed. Run: pip install openpyxl"}
        except Exception as e:
            return {"error": f"Failed to parse Excel file: {e}"}
    if ext in (".mp3", ".wav", ".m4a", ".mp4", ".png", ".jpg", ".jpeg", ".webp", ".heic"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            return {"error": "OpenAI API key required for audio transcription and image OCR."}
        from openai import OpenAI
        client = make_openai_client(api_key=api_key)
        try:
            if ext in (".mp3", ".wav", ".m4a", ".mp4"):
                with open(path, "rb") as audio_file:
                    return {"text": client.audio.transcriptions.create(model="whisper-1", file=audio_file).text}
            import base64, mimetypes
            if ext == ".heic":
                try:
                    import pillow_heif
                    from PIL import Image
                    pillow_heif.register_heif_opener()
                    image = Image.open(io.BytesIO(contents)).convert("RGB")
                    buf = io.BytesIO()
                    image.save(buf, format="JPEG")
                    contents = buf.getvalue()
                    mime_type = "image/jpeg"
                except ImportError:
                    return {"error": "HEIC library not installed. Run: pip install pillow-heif Pillow"}
            else:
                mime_type = mimetypes.guess_type(fname)[0] or f"image/{ext[1:]}"
            b64 = base64.b64encode(contents).decode("utf-8")
            resp = client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "Extract all text from this image exactly as written. If no text, describe the image briefly."},
                    {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                ]}],
                max_tokens=3000,
            )
            return {"text": resp.choices[0].message.content.strip()}
        except Exception as e:
            return {"error": f"Could not read the {'audio' if ext in ('.mp3', '.wav', '.m4a', '.mp4') else 'image'}: {e}"}
    return {"error": f"Unsupported file type '{ext}'. Supported: PDF, DOCX, TXT, XLSX, XLSM, XLAM, PNG, JPG, WEBP, HEIC, MP3, WAV, M4A, MP4."}


def analyze_document_uk(s3_key: str = None, filename: str = None) -> dict:
    """UK sibling of analyze_document. Returns the document's text plus UK analysis instructions, and no generated analysis:
    the UK persona may only state law it has fetched, so the model writes the analysis itself from fetched sources."""
    if not s3_key:
        return {"success": False, "error": "s3_key is required"}
    import s3_storage

    fname = filename or os.path.basename(s3_key)
    tmp_path = os.path.join(tempfile.gettempdir(), os.path.basename(s3_key))
    if not s3_storage.download_from_s3(s3_key, tmp_path):
        return {"success": False, "error": "File not found in S3 or S3 is not configured."}
    try:
        with open(tmp_path, "rb") as f:
            contents = f.read()
        if len(contents) > _UK_ANALYSIS_MAX_BYTES:
            return {"success": False, "error": "File too large. Maximum allowed size is 20MB."}

        extracted = _uk_extract_document_text(tmp_path, fname, contents)
        if extracted.get("error"):
            return {"success": False, "error": extracted["error"]}
        text = (extracted.get("text") or "").strip()
        if not text:
            return {"success": False, "error": "No text could be extracted from the document."}

        truncated = len(text) > _UK_ANALYSIS_CHAR_LIMIT
        return {
            "success": True,
            "filename": fname,
            "s3_key": s3_key,
            "char_count": len(text),
            "truncated": truncated,
            "document_text": text[:_UK_ANALYSIS_CHAR_LIMIT],
            "analysis_guide": _UK_ANALYSIS_GUIDE + (
                f" The text was cut at {_UK_ANALYSIS_CHAR_LIMIT:,} characters, so say the analysis covers only the start." if truncated else ""
            ),
        }
    except Exception as e:
        logging.error(f"[analyze_document_uk] Unexpected error: {e}")
        return {"success": False, "error": str(e)}
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def generate_legal_document(document_type: str, details: dict = None, format: str = "markdown", **kwargs) -> dict:
    """Generate a Philippine legal document based on type and provided details."""
    from openai import OpenAI

    if details is None:
        details = {}
    if kwargs:
        details.update(kwargs)

    doc_type_lower = document_type.lower().replace(" ", "_").replace("-", "_")
    if doc_type_lower not in DOCUMENT_TEMPLATES:
        return {
            "success": False,
            "error": f"Unknown document type: '{document_type}'",
            "available_types": list(DOCUMENT_TEMPLATES.keys()),
        }

    template = DOCUMENT_TEMPLATES[doc_type_lower]
    missing = [f for f in template["required_fields"] if not details.get(f)]
    if missing:
        return {
            "success": False,
            "error": "Missing required fields",
            "missing_fields": missing,
            "required_fields": template["required_fields"],
            "optional_fields": template.get("optional_fields", []),
        }

    filled = dict(details)
    for field in template.get("optional_fields", []):
        if field not in filled:
            filled[field] = "N/A"

    client = make_openai_client(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        prompt = template["prompt"].format(**filled)
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Philippine legal document drafter. Generate properly formatted legal documents "
                        "following Philippine legal conventions. Use proper legal document structure, formal language, "
                        "leave blanks (___) for missing info, and include proper signature/notary blocks."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return {
            "success": True,
            "document_type": doc_type_lower,
            "document_name": template["name"],
            "format": format,
            "content": response.choices[0].message.content,
            "fields_used": details,
            "disclaimer": "This is a template document generated by AI. Have it reviewed by a licensed attorney before use.",
            "next_steps": [
                "Review all details for accuracy",
                "Fill in any blank fields (marked with ___)",
                "Have a lawyer review the document",
                "If notarization is required, bring valid ID to a notary public",
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


def _format_authorities_block(authorities: list) -> tuple[str, str]:
    """Render a caller-supplied authorities list into a prompt section plus a citation instruction.

    `authorities` is expected to hold {"citation": ..., "proposition": ...} dicts already fetched by
    a research tool earlier in the conversation (see draft_pleading's docstring/manifest) — this only
    formats them, it never invents or looks anything up.
    """
    if not authorities:
        return "", (
            "No verified authorities were supplied for this draft — argue the grounds from the facts "
            "and general legal principles without citing a specific case or provision. Never invent a "
            "case name, citation, or statutory section from memory."
        )
    lines = []
    for a in authorities:
        if not isinstance(a, dict):
            continue
        citation = a.get("citation", "").strip()
        proposition = a.get("proposition", "").strip()
        if not citation:
            continue
        lines.append(f"- {citation}" + (f" — {proposition}" if proposition else ""))
    if not lines:
        return "", (
            "No verified authorities were supplied for this draft — argue the grounds from the facts "
            "and general legal principles without citing a specific case or provision. Never invent a "
            "case name, citation, or statutory section from memory."
        )
    block = "\n- Authorities (already verified — cite exactly as given, do not alter):\n" + "\n".join(lines)
    instruction = (
        "Weave the Authorities listed above into the argument to support the propositions they're paired "
        "with, citing each exactly as given. Do not cite any case, statute, or provision that is not in "
        "that list — if a point would benefit from an authority you don't have, argue it on the facts "
        "and general principles instead of naming one from memory."
    )
    return block, instruction


def draft_pleading(
    pleading_type: str,
    case_facts: str,
    grounds: str,
    relief_sought: str = None,
    responding_to: str = None,
    court: str = None,
    case_title: str = None,
    case_number: str = None,
    format: str = "markdown",
    authorities: list = None,
    **kwargs,
) -> dict:
    """Draft a complete Philippine court pleading in full prose.

    Deliberately free-form rather than a DOCUMENT_TEMPLATES-style fixed
    template: a pleading's substance is its legal argument, which varies
    per case and can't be reduced to filling blanks in a fixed paragraph
    shape (see generate_legal_document for the kind of document where that
    approach is correct — affidavits, contracts, and similar boilerplate).
    """
    from openai import OpenAI

    missing = [f for f, v in (("pleading_type", pleading_type), ("case_facts", case_facts), ("grounds", grounds)) if not v]
    if missing:
        return {
            "success": False,
            "error": "Missing required fields",
            "missing_fields": missing,
        }

    court = court or "___________________________"
    case_title = case_title or "___________________________"
    case_number = case_number or "___________________"
    responding_to_block = f"\n- Responding to: {responding_to}" if responding_to else ""
    relief_block = (
        f"\n- Relief sought: {relief_sought}" if relief_sought
        else "\n- Relief sought: not specified — infer the standard prayer appropriate to this pleading type from the grounds given."
    )
    authorities_block, citation_instruction = _format_authorities_block(authorities)

    prompt = f"""Draft a complete Philippine {pleading_type}, ready to file.

- Court: {court}
- Case: {case_title}
- Case No.: {case_number}{responding_to_block}
- Facts: {case_facts}
- Grounds: {grounds}{relief_block}{authorities_block}

Write the full pleading text now — not an outline, not a summary of what it should say. Expand the grounds above into properly developed legal argument in numbered paragraphs, applying the facts to the applicable law/rules. Include:
1. Caption block (court, parties, case number, pleading title, formatted as a PH pleading)
2. Prefatory statement identifying the pleading and what it responds to, if applicable
3. Body: numbered paragraphs developing each ground into full argument
4. Prayer/WHEREFORE clause requesting the relief sought
5. Verification and Certification Against Forum Shopping block (standard boilerplate, leave signature/notary details blank)
6. Signature block for counsel (leave name/roll number/address blank)

{citation_instruction}

Leave blanks (___) only for information that was not provided above — never leave the argument itself as a placeholder or description."""

    client = make_openai_client(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a Philippine litigation lawyer drafting a pleading for filing. "
                        "Always produce the complete, final document text — full sentences and "
                        "paragraphs a lawyer could file as-is. Never respond with a description, "
                        "outline, or explanation of what the pleading should contain instead of "
                        "writing it. Use proper Philippine pleading structure and formal legal "
                        "language throughout."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return {
            "success": True,
            "pleading_type": pleading_type,
            "format": format,
            "content": response.choices[0].message.content,
            "disclaimer": "This is an AI-drafted pleading. Have it reviewed by a licensed attorney before filing.",
            "next_steps": [
                "Review all details and arguments for accuracy",
                "Fill in any blank fields (marked with ___)",
                "Have a lawyer review the document before filing",
                "Confirm court, branch, and case number are correct",
            ],
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


# ---------------------------------------------------------------------------
# Cosmetics helpers
# ---------------------------------------------------------------------------

_CONCERN_SEARCH_TERMS = {
    "acne-prone":  "acne",
    "anti-aging":  "anti-aging",
    "brightening": "brightening",
    "dark circles": "dark",
}


def _http_get_json(url: str, headers: dict = None) -> dict:
    req = urllib.request.Request(url, headers=headers or {})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_post_json(url: str, body: dict, headers: dict = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={**(headers or {}), "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_patch_json(url: str, body: dict, headers: dict = None) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={**(headers or {}), "Content-Type": "application/json"}, method="PATCH")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


# Capped the same way get_outfits_by_category caps to 4 — a chat answer showing every match
# rather than a focused shortlist isn't useful, even once ranked.
_FIND_LAWYER_MAX_RESULTS = 5


def find_lawyer(case_areas_json: str, uk_jurisdiction: str = None, budget_min_minor: int = None, budget_max_minor: int = None) -> dict:
    """Fetch lawyers currently eligible to take this case from ilovelawyer-api, ranked by
    closeness to the client's budget. No AI selection in this tool — the eligibility rule and
    the budget ranking both live on the ilovelawyer-api side; this returns that result directly."""
    try:
        case_areas = json.loads(case_areas_json) if case_areas_json else []
    except (TypeError, ValueError):
        return {"success": False, "error": "invalid_case_areas", "message": "case_areas_json must be a JSON array of strings, e.g. '[\"EMPLOYMENT\"]'."}
    if not isinstance(case_areas, list) or not case_areas:
        return {"success": False, "error": "invalid_case_areas", "message": "Provide at least one case area."}

    base = (os.getenv("ILOVELAWYER_API_BASE") or "").rstrip("/")
    api_key = os.getenv("CHAT_WONDER_API_KEY", "")
    if not base:
        _logger.error("find_lawyer missing_config=ILOVELAWYER_API_BASE")
        return {"success": False, "error": "missing_config", "message": "ILOVELAWYER_API_BASE is not set in user_functions.env"}
    if not api_key:
        _logger.error("find_lawyer missing_config=CHAT_WONDER_API_KEY")
        return {"success": False, "error": "missing_config", "message": "CHAT_WONDER_API_KEY is not set in user_functions.env"}

    budget = None
    if budget_min_minor is not None or budget_max_minor is not None:
        budget = {"minMinor": budget_min_minor, "maxMinor": budget_max_minor, "feePreference": None}
    body = {"caseAreas": case_areas, "ukJurisdiction": uk_jurisdiction, "budget": budget}

    url = f"{base}/api/v1/lawyer-matching/by-criteria"
    t0 = time.perf_counter()
    try:
        data = _http_post_json(url, body, {"x-api-key": api_key})
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.error("find_lawyer http_%s elapsed=%.0fms error=%s", e.code, elapsed_ms, e)
        return {"success": False, "error": f"http_{e.code}", "message": f"find_lawyer failed: {e}"}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.error("find_lawyer failed elapsed=%.0fms error=%s", elapsed_ms, e)
        return {"success": False, "error": str(e), "message": f"find_lawyer failed: {e}"}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    matches = data.get("matches") or []
    ids = [m["lawyerId"] for m in matches if isinstance(m, dict) and m.get("lawyerId")][:_FIND_LAWYER_MAX_RESULTS]
    _logger.info("find_lawyer case_areas=%s uk_jurisdiction=%s matched=%d elapsed=%.0fms", case_areas, uk_jurisdiction, len(ids), elapsed_ms)

    if not ids:
        return {"success": True, "ids": [], "message": "No currently available lawyers match this case yet."}
    return {"success": True, "ids": ids}


def mark_case_ready() -> dict:
    """Call only once every must-have checklist item has been verified against the actual
    conversation — never on a guess. No arguments. This is intake's own self-report: the
    assistant's job is to call it, not to judge whether it will be acted on — ilovelawyer-api
    independently re-checks readiness (against Jev's own per-criterion assessment, already used
    for the "suggest a lawyer" offer) before treating the case as actually ready."""
    return {"success": True}


# Must match ilovelawyer-api's UK_JURISDICTIONS validation exactly (client-case.validation.ts).
_VALID_UK_JURISDICTIONS = {"England and Wales", "Scotland", "Northern Ireland"}
# Must match ilovelawyer-api's CLIENT_PARTY_ROLES exactly (client-case.constants.ts).
_VALID_PARTY_ROLES = {"Opposing party", "Witness", "Other"}


def create_case(
    title: str = None,
    case_areas_json: str = None,
    uk_jurisdiction: str = None,
    where: str = None,
    side: str = None,
    others_json: str = None,
    notes: str = None,
    outside_uk: bool = None,
    case_id: str = None,
    user_id: str = None,
) -> dict:
    """Write the case details drafted so far from the conversation onto the real case record in
    ilovelawyer-api. `case_id`/`user_id` are injected from the session, never supplied by the
    model — a missing one here means the injection didn't happen, not that the client left
    something out, so it's reported as a defensive guard rather than a user-facing failure.

    `outside_uk` is the only way to record "outside the UK": a case with no uk_jurisdiction is
    "not known yet" to ilovelawyer-api (#973), so leaving uk_jurisdiction out never says it."""
    if not case_id or not user_id:
        return {"success": False, "error": "missing_session_context", "message": "create_case requires case_id and user_id from the session; neither is a model-supplied parameter."}

    try:
        case_areas = json.loads(case_areas_json) if case_areas_json else None
    except (TypeError, ValueError):
        return {"success": False, "error": "invalid_case_areas", "message": "case_areas_json must be a JSON array of strings, e.g. '[\"EMPLOYMENT\"]'."}
    if case_areas is not None:
        if not isinstance(case_areas, list) or not all(isinstance(a, str) for a in case_areas):
            return {"success": False, "error": "invalid_case_areas", "message": "case_areas_json must be a JSON array of strings, e.g. '[\"EMPLOYMENT\"]'."}
        if not case_areas:
            case_areas = None
    try:
        others = json.loads(others_json) if others_json else None
    except (TypeError, ValueError):
        return {"success": False, "error": "invalid_others", "message": "others_json must be a JSON array of objects with name and role, e.g. '[{\"name\": \"Acme Ltd\", \"role\": \"Opposing party\"}]'."}
    if others is not None:
        if not isinstance(others, list) or not all(isinstance(o, dict) for o in others):
            return {"success": False, "error": "invalid_others", "message": "others_json must be a JSON array of objects with name and role, e.g. '[{\"name\": \"Acme Ltd\", \"role\": \"Opposing party\"}]'."}
        if not others:
            others = None
        else:
            # Same reasoning as the uk_jurisdiction guard below: a descriptive role the model
            # invents ("co-heir", "landlord") gets the whole request rejected over one word.
            # Normalize to the closest valid bucket rather than lose the party entirely.
            for party in others:
                if party.get("role") not in _VALID_PARTY_ROLES:
                    _logger.warning("create_case normalized invalid party role=%r to 'Other'", party.get("role"))
                    party["role"] = "Other"

    # ilovelawyer-api rejects the whole request if this isn't exactly one of these three (or
    # omitted) — the manifest enum should already keep the model on the rails, but a bad value
    # here would otherwise fail the entire write over one field. Drop it rather than risk that.
    if uk_jurisdiction is not None and uk_jurisdiction not in _VALID_UK_JURISDICTIONS:
        _logger.warning("create_case dropped invalid uk_jurisdiction=%r (not in %s)", uk_jurisdiction, sorted(_VALID_UK_JURISDICTIONS))
        uk_jurisdiction = None

    # Only ever sent as true: the AI has no business clearing "outside the UK" on its own, and a
    # real jurisdiction clears it server-side anyway. A jurisdiction wins over it — the API rejects
    # the pair outright, which would lose every other field in the same write. Models sometimes
    # send booleans as strings, so "true" counts too.
    outside_uk = outside_uk is True or (isinstance(outside_uk, str) and outside_uk.strip().lower() == "true")
    if outside_uk and uk_jurisdiction is not None:
        _logger.warning("create_case dropped outside_uk=true alongside uk_jurisdiction=%r", uk_jurisdiction)
        outside_uk = False

    base = (os.getenv("ILOVELAWYER_API_BASE") or "").rstrip("/")
    api_key = os.getenv("CHAT_WONDER_API_KEY", "")
    if not base:
        _logger.error("create_case missing_config=ILOVELAWYER_API_BASE")
        return {"success": False, "error": "missing_config", "message": "ILOVELAWYER_API_BASE is not set in user_functions.env"}
    if not api_key:
        _logger.error("create_case missing_config=CHAT_WONDER_API_KEY")
        return {"success": False, "error": "missing_config", "message": "CHAT_WONDER_API_KEY is not set in user_functions.env"}

    fields = {
        "title": title,
        "areas": case_areas,
        "ukJurisdiction": uk_jurisdiction,
        "outsideUk": True if outside_uk else None,
        "where": where,
        "side": side,
        "others": others,
        "notes": notes,
    }
    body = {"userId": user_id, **{k: v for k, v in fields.items() if v is not None}}

    url = f"{base}/api/v1/intake-cases/{case_id}"
    t0 = time.perf_counter()
    try:
        data = _http_patch_json(url, body, {"x-api-key": api_key})
    except urllib.error.HTTPError as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        try:
            body = e.read().decode("utf-8")
        except Exception:
            body = ""
        _logger.error("create_case http_%s elapsed=%.0fms error=%s body=%s", e.code, elapsed_ms, e, body)
        return {"success": False, "error": f"http_{e.code}", "message": f"create_case failed: {e} {body}"}
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _logger.error("create_case failed elapsed=%.0fms error=%s", elapsed_ms, e)
        return {"success": False, "error": str(e), "message": f"create_case failed: {e}"}
    elapsed_ms = (time.perf_counter() - t0) * 1000

    _logger.info("create_case case_id=%s user_id=%s elapsed=%.0fms", case_id, user_id, elapsed_ms)
    return {"success": True, "case": data}


