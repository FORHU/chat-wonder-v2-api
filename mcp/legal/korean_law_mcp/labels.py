"""Human-readable titles for Korean Law MCP results.

The Korean Law MCP (unlike juris.ph and the UK Legal MCP) returns plain formatted text for
every tool — no structured JSON, no "url"/"title" fields (confirmed live against
search_law/get_law_text/search_decisions). Without this module every pooled entry showed up
in the chat UI's Sources panel as "Untitled (no url)" even though real retrieval happened,
because collect_tool_result_urls()/allowed_url_set() (legal_citations.py) only recognize
specific JSON keys, which this MCP's text blobs never have.

There is also no stable per-article/per-case public URL to attach (law.go.kr has no documented
deep-link scheme the way legislation.gov.uk/TNA do) — see legal_prompt_kr.txt's citation
convention, which cites by name/number instead of a link. This module only derives a display
title; it deliberately does not invent a "url" field."""

from __future__ import annotations

import re
from typing import Optional

_NUMBERED_LINE = re.compile(r"^\d+\.\s*([^\n\[]+)", re.M)
_LAW_NAME_LINE = re.compile(r"법령명:\s*(.+)")
_ARTICLE_HEADING = re.compile(r"^(제\S+조(?:의\d+)?\s+\S.*)$", re.M)
_CASE_BRACKET_LINE = re.compile(r"^\[\d+\]\s*(.+)", re.M)
_CASE_NUMBER_LINE = re.compile(r"사건번호:\s*(.+)")
_LAW_NAME_SEARCH_TITLE = re.compile(r"법령명:\s*(.+?)(?:\n|$)")


def _first_line(text: str, limit: int = 80) -> str:
    for line in (text or "").splitlines():
        s = line.strip()
        if s:
            return s[:limit]
    return ""


def title_for(tool_name: str, func_args: Optional[dict], text: str) -> str:
    """Best-effort display title for one Korean Law MCP tool result's text blob."""
    text = text or ""
    func_args = func_args or {}

    if tool_name == "search_law":
        m = _NUMBERED_LINE.search(text)
        if m:
            return f"Korean Law Search: {m.group(1).strip()}"
        return f"Korean Law Search: {func_args.get('query', '')}".strip()

    if tool_name == "get_law_text":
        name_m = _LAW_NAME_LINE.search(text)
        heading_m = _ARTICLE_HEADING.search(text)
        name = name_m.group(1).strip() if name_m else ""
        heading = heading_m.group(1).strip() if heading_m else (func_args.get("jo") or "")
        if name and heading:
            return f"{name} {heading}"
        return name or heading or "Korean Law Text"

    if tool_name == "get_annexes":
        return f"Annex: {func_args.get('lawName', '')}".strip() or "Korean Law Annex"

    if tool_name == "ordinance_radar":
        name = func_args.get("ordinanceName") or func_args.get("query") or func_args.get("id") or ""
        return f"Ordinance Radar: {name}".strip() if name else "Ordinance Radar"

    if tool_name == "search_decisions":
        case_m = _CASE_BRACKET_LINE.search(text)
        number_m = _CASE_NUMBER_LINE.search(text)
        if case_m and number_m:
            return f"{case_m.group(1).strip()} ({number_m.group(1).strip()})"
        if case_m:
            return case_m.group(1).strip()
        return f"Decision Search: {func_args.get('query', '')}".strip() or "Decision Search"

    if tool_name == "get_decision_text":
        case_m = _CASE_BRACKET_LINE.search(text) or re.search(r"^(.+?)\n", text)
        number_m = _CASE_NUMBER_LINE.search(text)
        if case_m and number_m:
            return f"{case_m.group(1).strip()} ({number_m.group(1).strip()})"
        return _first_line(text) or "Korean Decision"

    if tool_name == "legal_research":
        task = func_args.get("task", "full_research")
        return f"Korean Legal Research ({task}): {func_args.get('query', '')}".strip()

    if tool_name == "legal_analysis":
        mode = func_args.get("mode", "")
        return f"Korean Legal Analysis ({mode})" if mode else "Korean Legal Analysis"

    return _first_line(text) or "Korean Law MCP Result"
