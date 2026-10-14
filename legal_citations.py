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


_URL_KEYS = ("url", "page_url", "pdf_url", "xml_url", "resolved_url")


def _tna_url_from_uri(uri: str) -> str:
    """TNA Find Case Law page URL for a bare judgment slug, e.g. 'ewca/civ/2024/1'."""
    return f"https://caselaw.nationalarchives.gov.uk/{uri.strip('/')}"


def collect_tool_result_urls(search_results) -> List[str]:
    """Unique absolute URLs from search/get tool rows, order preserved.

    Walks nested dicts/lists — tool payloads vary in shape across juris.ph
    (flat rows, `document.url`) and the UK Legal MCP (deeply nested Hansard
    contributions, bare `uri` slugs instead of URLs) — so every citable URL
    surfaces regardless of which tool produced it.
    """
    if not search_results:
        return []
    urls: List[str] = []
    seen: Set[str] = set()

    def add(raw):
        url = str(raw or "").strip()
        if url.startswith("http") and url not in seen:
            seen.add(url)
            urls.append(url)

    def walk(obj, depth=0):
        if depth > 6 or obj is None:
            return
        if isinstance(obj, dict):
            for key in _URL_KEYS:
                if obj.get(key):
                    add(obj[key])
            uri = obj.get("uri")
            if isinstance(uri, str) and uri and "://" not in uri:
                add(_tna_url_from_uri(uri))
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj[:50]:
                walk(item, depth + 1)

    for row in search_results:
        walk(row)
    return urls


def select_related_cases(search_results, limit: int = 6) -> List[dict]:
    """Dedupe the legal tool-result pool by document identity and rank it.

    get_case/get_republic_act entries are model-vetted (ADR-0002 requires a
    structured fetch before any holding statement), so they outrank raw,
    unvetted search rows for the same document. Within each tier, sort by the
    MCP's own relevance score (search rows only; vetted entries have none and
    keep tier order).
    """
    if not search_results:
        return []

    def identity(row: dict) -> str:
        meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        return (
            str(row.get("item_id") or row.get("id") or "").strip()
            or str(meta.get("id") or "").strip()
            or str(row.get("url") or meta.get("url") or "").strip()
        )

    best: dict = {}
    for row in search_results:
        if not isinstance(row, dict):
            continue
        key = identity(row)
        if not key:
            continue
        vetted = bool(row.get("document") or row.get("prefetched"))
        candidate = dict(row)
        candidate["_vetted"] = vetted
        existing = best.get(key)
        if existing is None or (vetted and not existing["_vetted"]):
            best[key] = candidate

    ranked = sorted(
        best.values(),
        key=lambda r: (not r["_vetted"], -(r.get("score") or 0.0)),
    )

    results = []
    for r in ranked[:limit]:
        meta = r.get("metadata") if isinstance(r.get("metadata"), dict) else {}
        results.append(
            {
                "type": r.get("type") or "legal_document",
                "title": r.get("title"),
                "url": r.get("url") or meta.get("url"),
                "case_number": r.get("case_number"),
                "ra_number": r.get("ra_number"),
                "year": r.get("year") or meta.get("year"),
                "snippet": r.get("snippet"),
                "relevance": r.get("score"),
                "vetted": r["_vetted"],
            }
        )
    return results


def _normalize_url(url: str) -> str:
    return str(url or "").strip().rstrip("/")


def allowed_url_set(search_results) -> Set[str]:
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


def collect_tool_result_text_corpus(search_results, extra_texts=None) -> str:
    """Flatten retrieved snippets / structured fields for quote verification.

    extra_texts: further text a quotation may legitimately come from — the case's own exhibits."""
    if not search_results and not extra_texts:
        return ""
    parts: List[str] = [t.strip() for t in (extra_texts or []) if isinstance(t, str) and t.strip()]

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

    for row in search_results or []:
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


def is_unverified_blockquote(body: str, corpus_norm: str) -> bool:
    """True when a blockquote body must be grounded in retrieved text but isn't.

    Shared by strip_unverified_blockquotes (rewrite) and legal_verify.audit_legal_draft
    (report) so both apply the same exemptions: lines carrying a link / legal-ref
    anchor and short bodies are never treated as unverified.
    """
    if re.search(r"\[.+\]\(.+\)", body) or re.search(r'class="legal-ref', body):
        return False
    bare = _normalize_quote_text(body)
    if len(bare) < 24:
        return False
    return not quote_appears_in_corpus(body, corpus_norm)


def strip_unverified_blockquotes(
    text: str,
    search_results,
    on_strip: OnEvent = None,
    extra_texts=None,
) -> str:
    """Demote blockquote lines whose body is not grounded in tool-result text (or, via
    extra_texts, the case's own exhibits)."""
    if not text or not isinstance(text, str):
        return text

    corpus = collect_tool_result_text_corpus(search_results, extra_texts)
    stripped = [0]

    def repl(m: re.Match) -> str:
        body = m.group("body")
        if not is_unverified_blockquote(body, corpus):
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


def classify_href(href: str, allowed: Set[str]) -> str:
    """Cite-gate verdict for one markdown href against the normalized allowed pool.

    Returns "skip" (not a gateable href — anchors, mailto, bare text), "ok",
    "truncated" (placeholder like `juris.ph/case/...`), or "unverified".
    Shared by gate_unverified_legal_urls (rewrite) and legal_verify (report).
    """
    h = (href or "").strip()
    if not _is_gateable_href(h):
        return "skip"
    if _is_truncated_href(h):
        return "truncated"
    if _normalize_url(h) in allowed:
        return "ok"
    return "unverified"


def gate_unverified_legal_urls(
    text: str,
    search_results,
    on_gate: OnEvent = None,
) -> str:
    """Demote markdown links whose href is not an exact URL from tool results."""
    if not text or not isinstance(text, str):
        return text

    allowed = allowed_url_set(search_results)
    gated = [0]

    def repl(m: re.Match) -> str:
        label, href = m.group(1), m.group(2)
        verdict = classify_href(href, allowed)
        if verdict in ("skip", "ok"):
            return m.group(0)
        gated[0] += 1
        return label

    out = _MD_LINK.sub(repl, text)
    if gated[0] and on_gate:
        on_gate("unverified_url")
    return out


_BUNDLE_REFS_IN_LABEL = re.compile(
    r"\[((?:D\d{2}[^\];]*;\s*)+)([^\]]+ (?:Law|Jurisprudence))\]\((https?://[^)]+)\)"
)


def split_bundle_refs_out_of_labels(text: str) -> str:
    """`[D01, paras 3-5; D06, paras 2-4; HSWA 1974, s 3 Law](url)` → the bundle references as
    plain text, then only the authority linked. The model sometimes packs a trailing string of
    case-document pinpoints into the same brackets as the authority; left alone, the link
    formatter turns the whole run into the anchor text of one statute link."""
    if not text or not isinstance(text, str):
        return text
    return _BUNDLE_REFS_IN_LABEL.sub(lambda m: f"[{m.group(1).rstrip()}] [{m.group(2)}]({m.group(3)})", text)


def format_legal_citation_links(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    text = split_bundle_refs_out_of_labels(text)
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


_LEGAL_REF_ANCHOR = re.compile(
    r'<a href="([^"]+)" class="legal-ref (law|jurisprudence)"[^>]*>(.*?)</a>', re.S
)
_CITATION_LABEL_SUFFIX = re.compile(r"\s+(?:Law|Jurisprudence)$")


def _citation_key(m: "re.Match") -> tuple:
    return (re.sub(r"\s+", " ", m.group(3)).strip().lower(), m.group(1))


def _drop_repeated_anchor(text: str, m: "re.Match") -> str:
    """Remove one repeated citation anchor with the least disruption to the sentence around it."""
    before, after = text[: m.start()], text[m.end() :]
    line_start = before.rfind("\n") + 1
    newline = after.find("\n")
    line_end = len(text) if newline < 0 else m.end() + newline  # index of the newline, or end

    # Followed by "; <next citation>": a trail, so drop this one and its separator.
    sep = re.match(r"[ \t]*;[ \t]*(?=<a )", after)
    if sep:
        return before + after[sep.end() :]

    # Ends its block (optionally before "." or ";"): drop it and the separator in front of it.
    tail = re.match(r"[ \t]*[.;]?[ \t]*(?=\n|$)", after)
    if tail:
        # An attribution or bullet line that was only the citation ("> - [Case Jurisprudence]",
        # "- [Act, s 1 Law]") has nothing left: delete the whole line.
        if re.fullmatch(r"[ \t]*(?:>[ \t]*)?(?:[-*][ \t]*)?", text[line_start : m.start()]):
            return text[:line_start] + text[line_end + 1 :]
        trimmed = re.sub(r"[ \t]*[;,][ \t]*$", "", before).rstrip(" \t")
        # Only delete a citation that follows a complete sentence or another citation; one that is
        # the grammatical object ("It follows from [link].") keeps its words below.
        if not trimmed or trimmed[-1] in ".?!:)\n" or trimmed.endswith("</a>"):
            if tail.group(0).strip() == "." and trimmed and trimmed[-1] not in ".?!:)\n":
                trimmed += "."
            return trimmed + after[tail.end() :]

    # Inside a sentence or table cell: keep the words, drop the repeated link.
    return before + _CITATION_LABEL_SUFFIX.sub("", m.group(3)) + after


def dedupe_repeated_citation_links(text: str) -> str:
    """Keep the first occurrence of each citation (same label and href) and drop later repeats.

    A bottom-line or closing list that re-links everything already cited reads as repetition.
    A different provision of the same document has a different label (its own pinpoint), so it
    is never treated as a repeat. Run after format_legal_citation_links (anchors are present)."""
    if not text or not isinstance(text, str) or "legal-ref" not in text:
        return text
    for _ in range(500):  # each pass removes one repeat; the bound only guards a runaway loop
        seen = set()
        for m in _LEGAL_REF_ANCHOR.finditer(text):
            key = _citation_key(m)
            if key in seen:
                text = _drop_repeated_anchor(text, m)
                break
            seen.add(key)
        else:
            break
    return text


def apply_legal_citation_pipeline(
    text: str,
    search_results,
    *,
    legal_mode: bool,
    on_repair: OnEvent = None,
    on_gate: OnEvent = None,
    on_strip_quote: OnEvent = None,
    extra_quote_sources=None,
) -> str:
    """Repair → (legal) strip bad quotes → (legal) gate URLs → (legal) HTML format."""
    out = repair_legal_source_links(text, search_results, on_repair=on_repair)
    if legal_mode:
        out = strip_unverified_blockquotes(
            out, search_results, on_strip=on_strip_quote, extra_texts=extra_quote_sources
        )
        out = gate_unverified_legal_urls(out, search_results, on_gate=on_gate)
        out = format_legal_citation_links(out)
    return out
