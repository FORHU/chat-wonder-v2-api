# -*- coding: utf-8 -*-
"""Tool-call execution for the model's function-calling loop. Extracted from
the_server.py (chat-wonder-v2-api#182).

execute_function_call dispatches by name against the_server's own globals,
because that is where _load_user_functions injects every loaded tool
(resources/functions/user_functions.py, the StyleOS/legal tool clusters, the
built-in execute_code tool) — there is no separate tool registry. Rather than
drag that whole loading mechanism over here too, this module reaches back
into the_server for the live function object (and the handful of other
server globals it touches: _context, calculate_tokens, the document-tool
constants). The import is done lazily inside each function, not at module
level, so this module stays importable on its own and so the_server's own
`from app.legal.runtime import execute_function_call` at the point this used
to be defined doesn't hit a half-initialized module on either side.
"""

from __future__ import annotations

import json
import logging
import os
from threading import Thread

from uk_legal_mcp import urls as uk_urls
from uk_legal_mcp.authorities import authorities_from_pool as uk_authorities_from_pool


def _derive_uk_pleading_authorities(session_id: str) -> list:
    import the_server as _srv

    state = _srv._context.sessions.get(session_id)
    return uk_authorities_from_pool(getattr(state, "last_search_legal_results", None))


def execute_function_call(function_call: dict, session_id: str = None):
    import the_server as _srv

    func_name = function_call.get("name")
    try:
        func_args = json.loads(function_call.get("arguments", "{}"))
    except Exception:
        func_args = {}

    if not isinstance(func_name, str) or not hasattr(_srv, func_name):
        logging.error(f"Function {func_name} not found in globals.")
        return None

    try:
        if func_name == "navigate_app" and session_id:
            func_args["session_id"] = session_id
        if func_name == "create_case" and session_id:
            _cc_state = _srv._context.sessions.get(session_id)
            func_args["case_id"] = getattr(_cc_state, "case_id", None)
            func_args["user_id"] = getattr(_cc_state, "user_id", None)
        if func_name == "get_case_document" and session_id:
            _cd_state = _srv._context.sessions.get(session_id)
            # The model is only ever shown a handle (F1, ...), never the real id — see
            # _build_case_document_injection. Resolve it back to the real id before anything
            # else touches case_document_id. A value that isn't a known handle (an older
            # session, a direct API caller, or a raw id a test passes) is left as-is.
            _raw_cd_id = str(func_args.get("case_document_id") or "")
            if _cd_state is not None:
                func_args["case_document_id"] = (getattr(_cd_state, "doc_id_by_handle", None) or {}).get(_raw_cd_id.upper(), _raw_cd_id)
            _cd_id = str(func_args.get("case_document_id") or "")
            _allowed = getattr(_cd_state, "allowed_case_document_ids", None) if _cd_state is not None else None
            if _allowed is not None and _cd_id not in _allowed:
                return {
                    "success": False,
                    "error": "out_of_scope",
                    "message": "Case document is not attached to this case.",
                }
            # BM25-rank raw_chunks before get_case_document's char-cap truncation, so on the
            # whole-document fallback path (no case_document_chunk_ids filter — see that
            # function's docstring) the cut falls on the least relevant chunks instead of
            # whichever happen to sit last by chunkIndex.
            _cd_query = getattr(_cd_state, "current_query", "") if _cd_state is not None else ""
            if _cd_query:
                func_args["query"] = _cd_query
        _cached_case_document = None
        if func_name == "get_case_document" and session_id:
            _cd_state = _srv._context.sessions.get(session_id)
            _cd_id = func_args.get("case_document_id")
            # Serve from cache only when the cache holds the WHOLE document and the model is
            # asking for the whole document. A cached relevance-filtered slice must not answer
            # a follow-up "read it in full" call, and a paged/filtered request must go to the
            # API even if the full text is cached (the model asked for a specific window).
            _wants_whole = not func_args.get("case_document_chunk_ids") and func_args.get("limit") is None
            if _cd_state is not None and _cd_id and _cd_id in _cd_state.case_document_cache and _wants_whole:
                _cached = _cd_state.case_document_cache[_cd_id]
                if not _cached.get("partial"):
                    _cached_case_document = _cached
        # Same-turn dedup for Korean Law MCP calls: the model has been observed calling
        # legal_research twice with the identical query in one turn (confirmed live,
        # 2026-11-11). Unlike juris.ph/the UK MCP there's no per-row id to dedupe the pool by
        # after the fact, and every call burns against the server's shared public quota (see
        # korean-mcp-limitations-briefing), so an identical (tool, args) pair this turn is
        # served from cache instead of re-calling the MCP. state.turn_kr_call_cache is reset
        # per turn (reason_loop/streaming_reason_loop), so a later turn still calls fresh.
        _KR_DEDUP_TOOLS = {
            "search_law", "get_law_text", "get_annexes", "ordinance_radar",
            "legal_research", "legal_analysis", "search_decisions", "get_decision_text",
            "discover_tools_kr", "execute_tool_kr",
        }
        _kr_cache = None
        _kr_dedup_key = None
        _kr_cached_result = None
        if func_name in _KR_DEDUP_TOOLS and session_id:
            _kr_state = _srv._context.sessions.get(session_id)
            _kr_cache = getattr(_kr_state, "turn_kr_call_cache", None) if _kr_state is not None else None
            if _kr_cache is not None:
                try:
                    _kr_dedup_key = (func_name, json.dumps(func_args, sort_keys=True, ensure_ascii=False))
                except TypeError:
                    _kr_dedup_key = None
                if _kr_dedup_key is not None and _kr_dedup_key in _kr_cache:
                    _kr_cached_result = _kr_cache[_kr_dedup_key]
                    logging.info("korean_law_mcp %s deduped (identical call already made this turn)", func_name)

        if _cached_case_document is not None:
            result = {"success": True, "partial": False, **_cached_case_document}
        elif _kr_cached_result is not None:
            result = _kr_cached_result
        else:
            if func_name == "draft_pleading_uk" and session_id and not func_args.get("authorities"):
                _derived = _derive_uk_pleading_authorities(session_id)
                if _derived:
                    func_args["authorities"] = _derived
            result = getattr(_srv, func_name)(**func_args)
            if _kr_dedup_key is not None and _kr_cache is not None:
                _kr_cache[_kr_dedup_key] = result
        _LEGAL_RESULT_TOOLS = {
            "search_jurisprudence",
            "search_republic_acts",
            "get_case",
            "get_republic_act",
        }
        # The pool persists for the session; bound it so a long consultation
        # doesn't grow the cite-gate/quote corpus without limit.
        _LEGAL_POOL_MAX = int(os.getenv("LEGAL_POOL_MAX", "300"))
        if func_name in _LEGAL_RESULT_TOOLS and session_id and isinstance(result, dict) and result.get("success"):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                if func_name.startswith("search_"):
                    # Accumulate, never overwrite (same posture as the UK pool below).
                    # Overwriting wiped every earlier search row AND every vetted get_*
                    # entry the moment the model ran one more search — confirmed live:
                    # the cite gate then demoted 17-20 legitimately retrieved citations
                    # per turn. Rows already in the pool (by id) are kept as-is so a
                    # vetted get_* entry is never replaced by a thinner search row.
                    existing = list(state.last_search_legal_results or [])
                    seen_ids = {
                        str(r.get("id") or r.get("item_id") or "")
                        for r in existing if isinstance(r, dict)
                    }
                    for row in result.get("results", []) or []:
                        if not isinstance(row, dict):
                            continue
                        rid = str(row.get("id") or row.get("item_id") or "")
                        if rid and rid in seen_ids:
                            continue
                        seen_ids.add(rid)
                        existing.append(row)
                    state.last_search_legal_results = existing[-_LEGAL_POOL_MAX:]
                else:
                    # Merge get_* into citation pool for URL repair / metadata
                    entry = {
                        "id": result.get("id") or result.get("item_id"),
                        "item_id": result.get("item_id") or result.get("id"),
                        "title": result.get("title"),
                        "url": result.get("url"),
                        "type": result.get("type"),
                        "year": result.get("year"),
                        "snippet": result.get("snippet") or "",
                        "disposition": result.get("disposition") or "",
                        "document": result.get("document"),
                    }
                    # Keep quote-verification corpus from structured get_* fields
                    doc = result.get("document") if isinstance(result.get("document"), dict) else {}
                    if doc:
                        entry["snippet"] = entry["snippet"] or doc.get("factual_background") or doc.get("summary") or ""
                        entry["disposition"] = entry["disposition"] or doc.get("final_disposition") or doc.get("disposition") or ""
                        entry["court_reasoning"] = doc.get("court_reasoning") or ""
                        entry["full_text"] = doc.get("full_text") or doc.get("text") or ""
                    existing = list(state.last_search_legal_results or [])
                    existing.append(entry)
                    state.last_search_legal_results = existing
        _UK_LEGAL_RESULT_TOOLS = {
            "case_law_search",
            "judgment_get_header",
            "judgment_get_index",
            "judgment_get_paragraph",
            "case_law_grep_judgment",
            "legislation_search",
            "legislation_get_toc",
            "legislation_get_section",
            "get_legal_recommendation_uk",
            "citations_parse",
            "citations_resolve",
            "citations_network",
            "citations_format_oscola",
            "parliament_search_hansard",
            "parliament_policy_position_summary",
            "parliament_find_member",
            "parliament_member_debates",
            "parliament_member_interests",
            "parliament_search_petitions",
            "parliament_get_debate_divisions",
            "parliament_get_debate_contributions",
            "parliament_lookup_by_column",
            "bills_search_bills",
            "bills_get_bill",
            "votes_search_divisions",
            "votes_get_division",
            "committees_search_committees",
            "committees_get_committee",
            "committees_search_evidence",
            "hmrc_get_vat_rate",
            "hmrc_check_mtd_status",
            "hmrc_search_guidance",
        }
        if func_name in _UK_LEGAL_RESULT_TOOLS and session_id and isinstance(result, dict) and result.get("success"):
            # UK Legal MCP result shapes vary wildly by tool (search rows,
            # single judgment/section objects, deeply-nested Hansard
            # contributions) — unlike juris.ph's clean search_*/get_* split.
            # Keep the whole per-call payload as one pool entry and let
            # collect_tool_result_urls() walk it recursively for citable
            # URLs, rather than unwrapping per tool shape. Always accumulate
            # (never overwrite) since a UK turn chains many tool calls.
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                # Give section fetches / resolved section citations a real
                # legislation.gov.uk section URL (the MCP returns none / a search
                # page) — mutates `result` so the model cites it and the pool holds it.
                if func_name == "legislation_get_section":
                    uk_urls.enrich_section_result(result, func_args if isinstance(func_args, dict) else {})
                elif func_name == "citations_resolve":
                    uk_urls.enrich_resolve_result(result, state.last_search_legal_results or [])
                entry = {k: v for k, v in result.items() if k != "success"}
                if entry:
                    existing = list(state.last_search_legal_results or [])
                    existing.append(entry)
                    state.last_search_legal_results = existing
        _KR_LEGAL_RESULT_TOOLS = {
            "search_law",
            "get_law_text",
            "get_annexes",
            "ordinance_radar",
            "legal_research",
            "legal_analysis",
            "search_decisions",
            "get_decision_text",
        }
        if func_name in _KR_LEGAL_RESULT_TOOLS and session_id and isinstance(result, dict) and result.get("success") and _kr_cached_result is None:
            # _kr_cached_result is None here means this was a fresh call, not a same-turn dedup
            # hit — a deduped call's result is already in the pool from its first call, so
            # pooling it again would just show the identical source twice in the Sources panel.
            # Korean Law MCP result shapes vary by tool just as much as the UK MCP's do
            # (law text, annex extracts, decision bodies, research composites) — same posture
            # as the UK pool above: keep the whole per-call payload as one pool entry and let
            # collect_tool_result_urls()/the quote corpus walk it recursively, always
            # accumulating rather than overwriting.
            #
            # Unlike juris.ph and the UK MCP, this server returns plain formatted text for every
            # tool, not structured JSON (confirmed live) -- so there is no "title"/"url" key for
            # _search_results_to_source_metadata to read, and every entry showed up in the chat
            # UI's Sources panel as "Untitled (no url)" despite real retrieval happening. There is
            # also no stable per-article/per-case public URL to attach (law.go.kr has no
            # documented deep-link scheme) -- see legal_prompt_kr.txt's citation convention, which
            # cites by name/number instead of a link. korean_law_mcp.labels derives a display
            # title only; "url" is deliberately left unset.
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                entry = {k: v for k, v in result.items() if k != "success"}
                if entry:
                    from korean_law_mcp.labels import title_for as _kr_title_for

                    entry["title"] = _kr_title_for(func_name, func_args if isinstance(func_args, dict) else {}, result.get("text") or "")
                    existing = list(state.last_search_legal_results or [])
                    existing.append(entry)
                    state.last_search_legal_results = existing
        if func_name == "get_outfits_by_category" and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                state.last_outfit_ids_result = result.get("ids", [])
        if func_name == "find_lawyer" and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                state.last_find_lawyer_result = result.get("ids", [])
            # The raw lawyer ids go to the client only via the [LAWYER_MATCHES] frame above
            # (state.last_find_lawyer_result) — nothing stops the model from reciting them back
            # to the client otherwise, since whatever is in `result` becomes part of its context.
            if "ids" in result:
                _count = len(result["ids"])
                result = {
                    "success": result.get("success", True),
                    "count": _count,
                    "message": result.get("message") or (f"Found {_count} matching lawyer(s); shown to the client directly." if _count else "No currently available lawyers match this case yet."),
                }
        if func_name == "mark_case_ready" and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                # This is intake's self-report only — ilovelawyer-api re-checks it against Jev's
                # independent per-criterion readiness assessment before acting on it.
                state.last_mark_case_ready = bool(result.get("success"))
        if func_name in ("recommend_cosmetics", "get_cosmetics_by_skin_type") and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                state.last_cosmetics_result = result
            if func_name == "get_cosmetics_by_skin_type":
                _bg_skin_type = func_args.get("skin_type", "")
                if _bg_skin_type:
                    _SKIN_OILINESS = {"DRY": 20, "OILY": 80, "COMBINATION": 45, "NORMAL": 60, "SENSITIVE": 60}
                    _oiliness = _SKIN_OILINESS.get(_bg_skin_type.upper(), 50)
                    _skin_json = json.dumps({"output": [{"type": "oiliness", "ui_score": _oiliness}]})
                    def _scl_cosmetics(skin_json=_skin_json, skin_type=_bg_skin_type):
                        try:
                            r = getattr(_srv, "recommend_cosmetics")(skin_analysis_json=skin_json, sets=1)
                            logging.info(f"[SCL_TRACE] recommend_cosmetics skin_type={skin_type} success={r.get('success')}")
                        except Exception as _e:
                            logging.warning(f"[SCL_TRACE] recommend_cosmetics failed: {_e}")
                    Thread(target=_scl_cosmetics, daemon=True).start()
        if func_name == "search_nearby_places" and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                if result.get("multi_location"):
                    state.last_maps_result.extend(result.get("results", []))
                else:
                    state.last_maps_result.append(result)
        if func_name == "navigate_app" and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                state.last_nav_result = result
        if func_name == "generate_outfit_image" and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                state.last_tailor_result = result
        if func_name in _srv._DOCUMENT_TOOLS and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None and result.get("success") and result.get("content"):
                state.last_generated_file_result = {
                    "content": result["content"],
                    "format": result.get("format", "docx"),
                    "document_type": result.get("document_type") or "pleading",
                    "document_name": result.get("document_name") or result.get("pleading_type"),
                }
                # The drafted text travels to ilovelawyer-api via last_generated_file_result above,
                # which renders it and swaps the real download URL into the `(#download)` link the
                # model writes below. The model only gets a note — with the full text in its
                # context it echoes the whole document into the chat reply instead of pointing at
                # the file.
                result = {k: v for k, v in result.items() if k != "content"}
                result["document_delivery"] = (
                    "The full document has been generated and saved as a downloadable file. Do NOT "
                    "reproduce, quote or paste the document text. Reply in a few short, natural sentences: "
                    "make the document's name a markdown link whose target is exactly `#download` "
                    "(the real URL is filled in automatically), e.g. \"Here is your [affidavit of loss](#download).\" "
                    "or \"Please review and fill out this [affidavit of loss](#download).\" Use exactly one such "
                    "link, and never write any other URL. Then briefly say what it covers and what the user "
                    "still needs to do (fill in blanks, notarize)."
                )
        if func_name == "get_case_document" and session_id and isinstance(result, dict):
            state = _srv._context.sessions.get(session_id)
            if state is not None:
                _cd_id = result.get("id") or func_args.get("case_document_id")
                if result.get("success") and result.get("text"):
                    entry = {
                        "id": _cd_id,
                        "name": result.get("name") or "",
                        "text": result["text"],
                        "chunk_count": result.get("chunk_count", 0),
                        "document_chunk_count": result.get("document_chunk_count", result.get("chunk_count", 0)),
                        "partial": bool(result.get("partial")),
                    }
                    entry["_token_count"] = _srv.calculate_tokens(entry["text"])
                    # A relevance-filtered re-read must never replace a full document already in
                    # the cache — the manifest would then advertise the exhibit as fetched while
                    # only a slice of it is in context (the Brackenmoor D01 §25 / D20.3 failure).
                    _prev = state.case_document_cache.get(_cd_id)
                    if _prev and not _prev.get("partial") and entry["partial"]:
                        entry = _prev
                    state.case_document_cache[_cd_id] = entry
                    active = [d for d in state.active_case_documents if d.get("id") != _cd_id]
                    active.append(entry)
                    # Evict oldest-first until the accumulated text fits a token budget, not a
                    # flat document-count cap — a count cap either wasted budget capping a
                    # handful of small documents at 10 when 30 would easily fit, or discarded
                    # whole documents once past 10 regardless of how little text each
                    # contributed. ilovelawyer-api's per-document chunk floor (see
                    # DocumentChunkRepo.findRelevantByCase) means most documents now contribute
                    # only a few chunks each, so a token budget scales far better than a count.
                    while len(active) > 1 and sum(d.get("_token_count", 0) for d in active) > _srv._CASE_DOCUMENT_TOKEN_BUDGET:
                        active.pop(0)
                    state.active_case_documents = active
                elif not result.get("success"):
                    state.document_fetch_error = {"case_document_id": _cd_id, "message": result.get("message") or result.get("error")}
        logging.info(f"Function {func_name} executed successfully.")
        return result
    except Exception as e:
        logging.error(f"Function {func_name} failed: {e}")
        return {"error": str(e)}
