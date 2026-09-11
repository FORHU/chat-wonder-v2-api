# -*- coding: utf-8 -*-
"""Verifier feedback loop for legal answers.

The cite-gate / quote-strip verifiers in legal_citations run *after* the model
has finished and silently rewrite the answer. This module runs the same checks
on the model's draft *inside* the tool-calling loop and turns failures into a
model-facing message, so the model gets one (configurable) chance to fix its
own citations before the finalizer gates whatever is left.

Pure helpers here have no server imports; make_legal_verifier is the only
piece that touches session state and metrics.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Set, Tuple

from legal_citations import (
    allowed_url_set,
    classify_href,
    collect_tool_result_text_corpus,
    is_unverified_blockquote,
)

_MD_LINK = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")
_BLOCKQUOTE_LINE = re.compile(r"^\s*>\s*(?P<body>.*)$", re.M)
_TITLE_KEYS = ("title", "name", "citation", "neutral_citation", "short_title", "case_name")
_URL_KEYS = ("url", "page_url", "pdf_url", "xml_url", "resolved_url")

@dataclass
class VerifierFeedback:
    """What a chain loop injects (message) and shows in the trace (summary)."""
    message: str
    summary: str


Verifier = Callable[[str, int], Optional[VerifierFeedback]]

# Control chunk yielded by the streaming chains when a rejected draft has already
# been yielded: the consumer (chat_stream) must drop its buffered draft. Follows
# the `__HITL__` control-chunk convention. Safe only because legal text is
# buffered server-side, never streamed live to the client.
DRAFT_DISCARD = "__DRAFT_DISCARD__"


@dataclass
class AuditReport:
    unverified_urls: List[Tuple[str, str]] = field(default_factory=list)
    truncated_urls: List[Tuple[str, str]] = field(default_factory=list)
    unverified_quotes: List[str] = field(default_factory=list)
    allowed: List[Tuple[str, str]] = field(default_factory=list)

    @property
    def has_issues(self) -> bool:
        return bool(self.unverified_urls or self.truncated_urls or self.unverified_quotes)

    @property
    def url_issue_count(self) -> int:
        return len(self.unverified_urls) + len(self.truncated_urls)

    def summary(self) -> str:
        parts = []
        if self.url_issue_count:
            parts.append(f"{self.url_issue_count} unverified citation link(s)")
        if self.unverified_quotes:
            parts.append(f"{len(self.unverified_quotes)} unverified quotation(s)")
        return ", ".join(parts) or "no issues"


def collect_tool_result_citables(search_results) -> List[Tuple[str, str]]:
    """(title, url) pairs from the tool-result pool, order preserved, deduped by URL.

    Mirrors legal_citations.collect_tool_result_urls' recursive walk but keeps the
    nearest title so the feedback can show the model which URL belongs to which
    authority — a mistyped href is then fixable without another tool call.
    """
    if not search_results:
        return []
    out: List[Tuple[str, str]] = []
    seen: Set[str] = set()

    def title_of(obj: dict) -> str:
        for key in _TITLE_KEYS:
            val = obj.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
        for key in _TITLE_KEYS:
            val = meta.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        return ""

    def walk(obj, depth=0):
        if depth > 6 or obj is None:
            return
        if isinstance(obj, dict):
            title = title_of(obj)
            for key in _URL_KEYS:
                url = str(obj.get(key) or "").strip()
                if url.startswith("http") and url not in seen:
                    seen.add(url)
                    out.append((title, url))
            for value in obj.values():
                walk(value, depth + 1)
        elif isinstance(obj, (list, tuple)):
            for item in obj[:50]:
                walk(item, depth + 1)

    for row in search_results:
        walk(row)
    return out


def audit_legal_draft(text: str, search_results) -> AuditReport:
    """Report (never rewrite) every citation link / blockquote the finalizer would demote."""
    report = AuditReport(allowed=collect_tool_result_citables(search_results))
    if not text or not isinstance(text, str):
        return report

    allowed = allowed_url_set(search_results)
    for m in _MD_LINK.finditer(text):
        label, href = m.group(1), m.group(2).strip()
        verdict = classify_href(href, allowed)
        if verdict == "truncated":
            report.truncated_urls.append((label, href))
        elif verdict == "unverified":
            report.unverified_urls.append((label, href))

    corpus = collect_tool_result_text_corpus(search_results)
    for m in _BLOCKQUOTE_LINE.finditer(text):
        body = m.group("body")
        if is_unverified_blockquote(body, corpus):
            report.unverified_quotes.append(body.strip())
    return report


def _clip(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def build_verifier_feedback(report: AuditReport, *, max_allowed: int = 30) -> str:
    """Model-facing self-check message listing what failed and how to fix it."""
    lines = ["[VERIFIER — self-check before delivery]"]
    lines.append(
        "Your draft answer failed automated citation checks. Fix ONLY the items below, "
        "then rewrite the complete answer."
    )

    if report.unverified_urls or report.truncated_urls:
        lines.append("")
        lines.append("Citation links whose URL is NOT among the sources retrieved this turn:")
        for label, href in report.truncated_urls:
            lines.append(f"- [{_clip(label, 80)}]({_clip(href, 120)}) — placeholder/truncated URL")
        for label, href in report.unverified_urls:
            lines.append(f"- [{_clip(label, 80)}]({_clip(href, 120)})")
        lines.append(
            "For each, in this order of preference: (a) if it is the same authority as one in "
            "the retrieved sources list below, replace the href with that exact URL; "
            "(b) otherwise RETRIEVE the authority now (PH: get_case / get_republic_act; "
            "UK: legislation_get_section, judgment_get_header, citations_resolve) and use the "
            "URL the tool returns — this is preferred over removing a citation the answer "
            "relies on; (c) only if it cannot be retrieved, drop the hyperlink and keep the "
            "plain citation text, and soften any holding that depended on it. "
            "Never invent or guess a URL."
        )

    if report.unverified_quotes:
        lines.append("")
        lines.append("Blockquotes whose wording was NOT found in any retrieved case/legislation text:")
        for q in report.unverified_quotes:
            lines.append(f"- > {_clip(q, 160)}")
        lines.append(
            "For each: either fetch the passage and quote it verbatim (PH: get_case with "
            "full text; UK: judgment_get_paragraph / legislation_get_section), or rewrite "
            "it as a paraphrase without blockquote formatting."
        )

    if report.allowed:
        lines.append("")
        lines.append("Retrieved sources this turn (title — URL):")
        for title, url in report.allowed[:max_allowed]:
            lines.append(f"- {_clip(title, 90) or '(untitled)'} — {url}")
        if len(report.allowed) > max_allowed:
            lines.append(f"- … and {len(report.allowed) - max_allowed} more")

    lines.append("")
    lines.append(
        "Rewrite the complete answer with these corrections applied. Keep everything else "
        "unchanged. Do not mention this check, the verifier, or that anything was revised."
    )
    return "\n".join(lines)


def inject_verifier_feedback(items: list, draft: str, feedback: VerifierFeedback) -> None:
    """Append the rejected draft + verifier message to a chain's message list.

    Both Chat Completions `messages` and Responses `input_items` accept plain
    role/content dicts (the loops already append `[Constraints]` system items
    this way), so one helper serves all four chain loops.
    """
    items.append({"role": "assistant", "content": draft})
    items.append({"role": "system", "content": feedback.message})


def verifier_trace_text(feedback: VerifierFeedback, round_no: int) -> Tuple[str, str]:
    """(text, summary) for broadcast_trace when a revise round starts."""
    return (
        f"Self-check round {round_no}: {feedback.summary} — revising",
        f"Before delivering, the AI checked its own citations against the sources it actually "
        f"retrieved and found {feedback.summary}. It is revising the answer to fix them.",
    )


def verify_max_rounds_from_env() -> int:
    """LEGAL_VERIFY_MAX_ROUNDS: 0 disables the loop (rollback lever); default 1."""
    try:
        return max(0, int(os.getenv("LEGAL_VERIFY_MAX_ROUNDS", "1")))
    except ValueError:
        return 1


def make_legal_verifier(state, *, max_rounds: Optional[int] = None) -> Optional[Verifier]:
    """Closure the chain loops call with (draft_text, rounds_done) when the model
    stops calling tools. Returns feedback text to inject, or None to accept the
    draft. Reads state.last_search_legal_results at call time because the pool
    grows during the turn (a revise round may retrieve more authorities)."""
    rounds = verify_max_rounds_from_env() if max_rounds is None else max_rounds
    if rounds <= 0:
        return None

    def verify(draft: str, rounds_done: int) -> Optional[VerifierFeedback]:
        from the_server import increment_metric_counter

        report = audit_legal_draft(draft, getattr(state, "last_search_legal_results", None))
        tags = {"round": str(rounds_done), "urls": str(report.url_issue_count), "quotes": str(len(report.unverified_quotes))}
        if not report.has_issues:
            if rounds_done > 0:
                logging.info("[legal-verify] revise round %d resolved all issues", rounds_done)
                increment_metric_counter("legal.verify_refine.resolved", value=1, tags=tags, session_id=None)
            return None
        if rounds_done >= rounds:
            logging.info("[legal-verify] %s remain after %d round(s); finalizer will gate", report.summary(), rounds_done)
            increment_metric_counter("legal.verify_refine.exhausted", value=1, tags=tags, session_id=None)
            return None
        logging.info("[legal-verify] round %d: %s — sending feedback", rounds_done + 1, report.summary())
        increment_metric_counter("legal.verify_refine.count", value=1, tags=tags, session_id=None)
        return VerifierFeedback(message=build_verifier_feedback(report), summary=report.summary())

    return verify
