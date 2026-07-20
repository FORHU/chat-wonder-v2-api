# -*- coding: utf-8 -*-
"""Pure helpers for legal citation repair, cite-gating, and quote verification."""

from __future__ import annotations

import re
from typing import Callable, List, Optional, Set

OnEvent = Optional[Callable[[str], None]]

_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BROKEN_SOURCES = re.compile(r"/sources/([^)\s\"\]]*)")
_TRUNCATION_MARKERS = ("...", "…", "[…]", "[...]")
_BLOCKQUOTE_LINE = re.compile(r"^(?P<prefix>\s*>\s*)(?P<body>.*)$", re.M)
_QUOTE_CHARS = re.compile(r'[\"“”‘’*_`]')


def collect_tool_result_urls(search_results) -> List[str]:
    """Unique absolute URLs from search/get tool rows, order preserved."""
    if not search_results:
        return []
    urls: List[str] = []
    seen: Set[str] = set()
    for r in search_results:
        if not isinstance(r, dict):
            continue
        candidates = [
            r.get("url"),
            (r.get("metadata") or {}).get("url") if isinstance(r.get("metadata"), dict) else None,
        ]
        doc = r.get("document")
        if isinstance(doc, dict):
            candidates.extend([doc.get("url"), doc.get("page_url"), doc.get("pdf_url")])
        for raw in candidates:
            url = str(raw or "").strip()
            if url.startswith("http") and url not in seen:
                seen.add(url)
                urls.append(url)
    return urls


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def _allowed_set(search_results) -> Set[str]:
    return {_normalize_url(u) for u in collect_tool_result_urls(search_results)}


def _is_truncated_href(href: str) -> bool:
    h = href.strip()
    return any(m in h for m in _TRUNCATION_MARKERS)


def _is_gateable_href(href: str) -> bool:
    h = href.strip()
    if not h:
        return False
    if h.startswith("/sources/"):
        return True
    if h.startswith("http://") or h.startswith("https://"):
        return True
    return False


def _normalize_quote_text(text: str) -> str:
    t = _QUOTE_CHARS.sub("", text or "")
    t = re.sub(r"\s+", " ", t).strip().lower()
    return t


def collect_tool_result_text_corpus(search_results) -> str:
    """Flatten retrieved snippets / structured fields for quote verification."""
    if not search_results:
        return ""
    parts: List[str] = []

    def walk(obj, depth=0):
        if depth > 6 or obj is None:
            return
        if isinstance(obj, str):
            s = obj.strip()
            if len(s) >= 20:
                parts.append(s)
            return
        if isinstance(obj, dict):
            for k, v in obj.items():
                if str(k).lower() in {"url", "pdf_url", "page_url", "id", "item_id", "score", "tags"}:
                    continue
                walk(v, depth + 1)
            return
        if isinstance(obj, (list, tuple)):
            for item in obj[:50]:
                walk(item, depth + 1)

    for row in search_results:
        walk(row)
    return _normalize_quote_text("\n".join(parts))


def quote_appears_in_corpus(quote_body: str, corpus_norm: str, *, min_len: int = 24) -> bool:
    """True if normalized quote (or a long contiguous chunk) appears in corpus."""
    q = _normalize_quote_text(quote_body)
    if len(q) < min_len:
        return True
    if not corpus_norm:
        return False
    if q in corpus_norm:
        return True
    window = min(48, len(q))
    if window < min_len:
        return False
    for i in range(0, len(q) - window + 1, 8):
        if q[i : i + window] in corpus_norm:
            return True
    return False


def strip_unverified_blockquotes(
    text: str,
    search_results,
    on_strip: OnEvent = None,
) -> str:
    """Demote blockquote lines whose body is not grounded in tool-result text."""
    if not text or not isinstance(text, str):
        return text

    corpus = collect_tool_result_text_corpus(search_results)
    stripped = [0]

    def repl(m: re.Match) -> str:
        body = m.group("body")
        if re.search(r"\[.+\]\(.+\)", body) or re.search(r'class="legal-ref', body):
            return m.group(0)
        bare = _normalize_quote_text(body)
        if len(bare) < 24:
            return m.group(0)
        if quote_appears_in_corpus(body, corpus):
            return m.group(0)
        stripped[0] += 1
        return (
            "*[Unverified quotation removed — wording was not found in retrieved "
            "case/RA text. Rely on the paraphrase and citation above/below.]*"
        )

    out = _BLOCKQUOTE_LINE.sub(repl, text)
    if stripped[0] and on_strip:
        on_strip("unverified_blockquote")
    return out


def repair_legal_source_links(
    text: str,
    search_results,
    on_repair: OnEvent = None,
) -> str:
    """Rewrite leftover /sources/... links to juris.ph URLs from the tool-result pool."""
    if not text or not isinstance(text, str) or not search_results:
        return text

    source_urls = collect_tool_result_urls(search_results)
    if not source_urls:
        return text

    idx = [0]

    def repl(m):
        replacement = source_urls[min(idx[0], len(source_urls) - 1)]
        idx[0] += 1
        return replacement

    repaired = _BROKEN_SOURCES.sub(repl, text)
    if repaired != text and on_repair:
        on_repair("sources_to_juris_url")
    return repaired


def gate_unverified_legal_urls(
    text: str,
    search_results,
    on_gate: OnEvent = None,
) -> str:
    """Demote markdown links whose href is not an exact URL from tool results."""
    if not text or not isinstance(text, str):
        return text

    allowed = _allowed_set(search_results)
    gated = [0]

    def repl(m: re.Match) -> str:
        label, href = m.group(1), m.group(2).strip()
        if not _is_gateable_href(href):
            return m.group(0)
        if _is_truncated_href(href):
            gated[0] += 1
            return label
        norm = _normalize_url(href)
        if norm in allowed:
            return m.group(0)
        gated[0] += 1
        return label

    out = _MD_LINK.sub(repl, text)
    if gated[0] and on_gate:
        on_gate("unverified_url")
    return out


def format_legal_citation_links(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    url_pattern = r"(https?://[^)]+|[^)]+)"

    def repl_law(m):
        return f'<a href="{m.group(2)}" class="legal-ref law" target="_blank">{m.group(1)}</a>'

    def repl_juris(m):
        return (
            f'<a href="{m.group(2)}" class="legal-ref jurisprudence" '
            f'target="_blank">{m.group(1)}</a>'
        )

    text = re.sub(r"\[([^\]]+ Law)\]\(" + url_pattern + r"\)", repl_law, text)
    text = re.sub(r"\[([^\]]+ Jurisprudence)\]\(" + url_pattern + r"\)", repl_juris, text)
    return text


def apply_legal_citation_pipeline(
    text: str,
    search_results,
    *,
    legal_mode: bool,
    on_repair: OnEvent = None,
    on_gate: OnEvent = None,
    on_strip_quote: OnEvent = None,
) -> str:
    """Repair → (legal) strip bad quotes → (legal) gate URLs → (legal) HTML format."""
    out = repair_legal_source_links(text, search_results, on_repair=on_repair)
    if legal_mode:
        out = strip_unverified_blockquotes(out, search_results, on_strip=on_strip_quote)
        out = gate_unverified_legal_urls(out, search_results, on_gate=on_gate)
        out = format_legal_citation_links(out)
    return out
