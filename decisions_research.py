# -*- coding: utf-8 -*-
"""Faster legal research with OpenAI's Decisions API -- prototype of step R2, "pick what to read".

Today a UK legal turn spends most of its loop deciding, one model call at a time, which section to
fetch next (legislation_get_section was the most frequent step in the 10 Nov 2026 logs, and 6 of 11
turns hit the 12-loop limit). This module makes those reading decisions with the Decisions API
(POST /v1/decisions: typed answers with probabilities, many questions per request, cheap and fast)
and fetches the chosen sections in parallel, attaching them to the search result the model is about
to read -- so it can cite them straight away instead of spending a loop on each fetch.

Flow, after a legislation search result (get_legal_recommendation_uk / legislation_search):
  1. Decision 1 -- per Act: how relevant is it to the question? (choice: central / related / irrelevant)
  2. Table of contents of the top Acts (parallel), then
     Decision 2 -- per section: would reading it help answer the question? (predicate)
  3. Fetch the top sections in parallel through the SAME execute_function_call the model's own tool
     calls go through, so they land in the source pool exactly as a model fetch would (citation
     gate, Confidence view and related cases all see them).
  4. The loop then hands each one to the model as a real legislation_get_section call + result in
     its history -- not tucked inside the search result. The UK prompt tells the model to fetch a
     section with legislation_get_section before stating what it says; in the first test (sections
     attached to the search result) it re-fetched them anyway and linked to them less.

Off unless LEGAL_DECISIONS_PREFETCH=true. Fails open everywhere: any error, timeout or empty answer
means "no prefetch" and the turn runs exactly as before. Case law is not prefetched yet (the UK
Legal MCP's case-law endpoint was timing out when this was written).
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

DECISIONS_MODEL = "gpt-6-luna"
DECISIONS_TIMEOUT_S = 20.0
MAX_ACTS_CONSIDERED = 8
MAX_ACTS_OPENED = 2
MAX_SECTIONS_JUDGED = 380  # the API allows 400 questions per request
MAX_SECTIONS_FETCHED = 5
TOC_PAGE = 200
TOC_MAX_PAGES = 4
SECTION_CHARS = 6000
# Provisional thresholds -- to be set from the R1/R2 comparison, not trusted as-is.
ACT_CENTRAL_MIN = 0.5
SECTION_HELPS_MIN = 0.5

LEGISLATION_TOOLS = {"get_legal_recommendation_uk", "legislation_search"}
_SECTION_ITEM = re.compile(r"^section-([0-9A-Za-z]+):\s*(.*)$")
_HEADING_ITEM = re.compile(r"^(part-[^:]*|chapter-[^:]*|.*crossheading[^:]*):\s*(.*)$")

RunTool = Callable[[str, dict], Optional[dict]]


def prefetch_enabled() -> bool:
    return os.getenv("LEGAL_DECISIONS_PREFETCH", "false").strip().lower() in ("1", "true", "yes", "on")


def decide(client, input_text: str, questions: List[dict]) -> Dict[str, dict]:
    """One Decisions API request; answers keyed by question name. Uses the SDK's generic post because
    the pinned openai version predates the `decisions` resource. Raises on failure (callers fail open)."""
    body = {"model": DECISIONS_MODEL, "input": input_text, "questions": questions}
    resp = client.with_options(timeout=DECISIONS_TIMEOUT_S).post("/decisions", body=body, cast_to=httpx.Response)
    data = resp.json()
    return {a.get("name"): a for a in (data.get("answers") or []) if isinstance(a, dict)}


def _acts_from(tool_name: str, result: dict) -> List[dict]:
    rows = result.get("legislation") if tool_name == "get_legal_recommendation_uk" else result.get("results")
    acts, seen = [], set()
    for r in rows or []:
        if not isinstance(r, dict):
            continue
        key = (r.get("type"), r.get("year"), r.get("number"))
        if not all(key) or key in seen:
            continue
        seen.add(key)
        acts.append({"title": str(r.get("title") or ""), "type": r["type"], "year": r["year"], "number": r["number"]})
        if len(acts) >= MAX_ACTS_CONSIDERED:
            break
    return acts


def _rank_acts(client, query: str, acts: List[dict]) -> List[dict]:
    questions = [{
        "name": f"act{i}",
        "type": "choice",
        "instructions": f'Legislation: "{a["title"]}" ({a["type"]} {a["year"]}/{a["number"]}).\n\n'
                        "How relevant is this legislation to answering the legal question in the input?",
        "choices": [
            {"value": "central", "description": "It contains the rules the question turns on."},
            {"value": "related", "description": "It touches the topic but is not where the answer is."},
            {"value": "irrelevant", "description": "It does not bear on the question."},
        ],
    } for i, a in enumerate(acts)]
    answers = decide(client, f"Legal question: {query}", questions)
    scored = []
    for i, a in enumerate(acts):
        probs = {p.get("value"): p.get("probability", 0.0) for p in (answers.get(f"act{i}") or {}).get("probabilities") or []}
        if probs.get("central", 0.0) >= ACT_CENTRAL_MIN:
            scored.append((probs["central"], a))
    scored.sort(key=lambda t: -t[0])
    return [a for _, a in scored[:MAX_ACTS_OPENED]]


def _toc_sections(act: dict, run_tool: RunTool) -> List[dict]:
    """Every section of one Act with the heading it sits under, paging the table of contents."""
    base = {"type": act["type"], "year": act["year"], "number": act["number"]}
    first = run_tool("legislation_get_toc", {**base, "offset": 0, "limit": TOC_PAGE}) or {}
    pages = [first]
    total = int(first.get("total_items") or 0)
    offsets = [o for o in range(TOC_PAGE, min(total, TOC_PAGE * TOC_MAX_PAGES), TOC_PAGE)]
    if offsets:
        with ThreadPoolExecutor(max_workers=len(offsets)) as pool:
            pages += list(pool.map(lambda o: run_tool("legislation_get_toc", {**base, "offset": o, "limit": TOC_PAGE}) or {}, offsets))
    sections, heading = [], ""
    for page in pages:
        for item in page.get("items") or []:
            item = str(item)
            m = _SECTION_ITEM.match(item)
            if m:
                sections.append({**act, "section": m.group(1), "heading": m.group(2).strip(), "under": heading})
                continue
            h = _HEADING_ITEM.match(item)
            if h:
                heading = h.group(2).strip()
    return sections


def _pick_sections(client, query: str, sections: List[dict]) -> List[dict]:
    sections = sections[:MAX_SECTIONS_JUDGED]
    questions = [{
        "name": f"s{i}",
        "type": "predicate",
        "instructions": f'{s["title"]}, section {s["section"]}: "{s["heading"]}"'
                        + (f' (under "{s["under"]}")' if s["under"] else "")
                        + ".\n\nWould reading this section help answer the legal question in the input?",
    } for i, s in enumerate(sections)]
    answers = decide(client, f"Legal question: {query}", questions)
    scored = []
    for i, s in enumerate(sections):
        p = (answers.get(f"s{i}") or {}).get("probability")
        if isinstance(p, (int, float)) and p >= SECTION_HELPS_MIN:
            scored.append((p, s))
    scored.sort(key=lambda t: -t[0])
    return [s for _, s in scored[:MAX_SECTIONS_FETCHED]]


def section_key(args: Any) -> Optional[Tuple[str, str, str, str]]:
    """(type, year, number, section) for a legislation_get_section call, ignoring max_chars, so a
    model request for an already-prefetched section is recognised however it formats the args."""
    if not isinstance(args, dict):
        return None
    parts = tuple(str(args.get(k) or "").strip().lower() for k in ("type", "year", "number", "section"))
    return parts if all(parts) else None


def prefetch_sections(tool_name: str, result: Any, query: str, client, run_tool: RunTool) -> Optional[dict]:
    """After a legislation search result: decide which sections to read and fetch them in parallel.
    Returns {"metrics": {...}, "fetched": [{"args": {...}, "result": {...}}, ...]}, or None when
    nothing was done. Does not modify `result`. Never raises."""
    if tool_name not in LEGISLATION_TOOLS or not isinstance(result, dict) or not query or client is None:
        return None
    t0 = time.perf_counter()
    try:
        acts = _acts_from(tool_name, result)
        if not acts:
            return None
        top_acts = _rank_acts(client, query, acts)
        if not top_acts:
            return {"metrics": {"acts": len(acts), "opened": 0, "fetched": 0,
                                "ms": int((time.perf_counter() - t0) * 1000)}, "fetched": []}
        with ThreadPoolExecutor(max_workers=len(top_acts)) as pool:
            sections = [s for secs in pool.map(lambda a: _toc_sections(a, run_tool), top_acts) for s in secs]
        chosen = _pick_sections(client, query, sections) if sections else []
        fetched: List[dict] = []
        if chosen:
            def fetch(s: dict) -> Optional[dict]:
                args = {"type": s["type"], "year": s["year"], "number": s["number"],
                        "section": s["section"], "max_chars": SECTION_CHARS}
                res = run_tool("legislation_get_section", args)
                if not isinstance(res, dict) or not res.get("success"):
                    return None
                return {"args": args, "result": res, "label": f'{s["title"]} s{s["section"]}'}
            with ThreadPoolExecutor(max_workers=len(chosen)) as pool:
                fetched = [f for f in pool.map(fetch, chosen) if f]
        metrics = {"acts": len(acts), "opened": len(top_acts), "judged": min(len(sections), MAX_SECTIONS_JUDGED),
                   "fetched": len(fetched), "sections": [f["label"] for f in fetched],
                   "ms": int((time.perf_counter() - t0) * 1000)}
        logging.info("decisions prefetch: %s", json.dumps(metrics, ensure_ascii=False))
        return {"metrics": metrics, "fetched": fetched}
    except Exception as e:
        logging.warning("decisions prefetch failed after %.2fs (turn continues unchanged): %s", time.perf_counter() - t0, e)
        return None
