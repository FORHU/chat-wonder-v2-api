# -*- coding: utf-8 -*-
"""Post-generation finalization for legal answers: the citation repair/gate
pipeline, doctrine guards, case-document-id redaction, and the intake variant
of the citation gate (no verifier/refine loop, no doctrine guards). Extracted
from the_server.py (chat-wonder-v2-api#182).
"""

from __future__ import annotations

import logging
import re

from legal_citations import apply_legal_citation_pipeline
from exhibit_quotes import exhibit_texts, unquote_exhibit_misquotes
from legal_fact_boost import (
    append_critical_doctrine_guards,
    append_missing_prefetched_mentions,
    append_uk_doctrine_guards,
    case_forum_text,
)
from app.core.tracing import increment_metric_counter


def _search_results_to_source_metadata(results: list) -> list:
    return [
        {
            "type": r.get("type") or "legal_document",
            "item_id": str(r.get("item_id") or r.get("id") or "").strip() or None,
            "title": r.get("title"),
            "category": r.get("type"),
            "year": r.get("year") or (r.get("metadata") or {}).get("year"),
            "source_url": r.get("url") or (r.get("metadata") or {}).get("url"),
            "snippet": r.get("snippet"),
            "relevance": r.get("score", 1.0),
        }
        for r in results
    ]


def _finalize_legal_citations(
    text: str, search_results, *, legal_mode: bool, extra_quote_sources=None, quotation_blocks_only: bool = False
) -> str:
    def on_repair(reason: str):
        logging.info("[legal-citation] repaired /sources/ link(s)")
        increment_metric_counter("legal.citation_repair.count", value=1, tags={"reason": reason}, session_id=None)

    def on_gate(reason: str):
        logging.info("[legal-citation] demoted unverified citation URL(s)")
        increment_metric_counter("legal.citation_gate.count", value=1, tags={"reason": reason}, session_id=None)

    def on_strip_quote(reason: str):
        logging.info("[legal-citation] removed unverified blockquote(s)")
        increment_metric_counter("legal.citation_quote_strip.count", value=1, tags={"reason": reason}, session_id=None)

    return apply_legal_citation_pipeline(
        text,
        search_results,
        legal_mode=legal_mode,
        on_repair=on_repair,
        on_gate=on_gate,
        on_strip_quote=on_strip_quote,
        extra_quote_sources=extra_quote_sources,
        quotation_blocks_only=quotation_blocks_only,
    )


def _redact_document_ids(text: str, state) -> str:
    """Strip anything id-shaped for a case document out of text the user or the persisted
    transcript will see: a raw case-document UUID (belt-and-braces — the model is never shown
    one, see _build_case_document_injection, but an older session or a model that ignores the
    instruction could still echo one) and a bracketed internal handle like [F1]. Both are
    replaced with the document's real name. A bare handle with no brackets is left alone — it
    could be real user text, not a reference."""
    if not text or not state:
        return text
    manifest = getattr(state, "case_document_manifest", None) or []
    handles = getattr(state, "doc_handle_by_id", None) or {}
    for m in manifest:
        doc_id = str(m.get("id") or "")
        if not doc_id:
            continue
        name = m.get("name") or "Document"
        if doc_id in text:
            text = text.replace(doc_id, name)
        handle = handles.get(doc_id)
        if handle:
            text = re.sub(rf"\[{re.escape(handle)}\]", name, text)
    return text


def _legal_jurisdiction_for_addendum(addendum_override) -> str:
    """Which jurisdiction's addendum is active, from its distinguishing header line.

    Used everywhere a call site only has the addendum text in hand (not the persona string)
    to pick the right _finalize_legal_response(jurisdiction=...) branch."""
    if addendum_override and "UK LEGAL ASSISTANT MODE" in addendum_override:
        return "UK"
    if addendum_override and "KOREAN LEGAL ASSISTANT MODE" in addendum_override:
        return "KR"
    return "PH"


def _finalize_legal_response(
    text: str,
    search_results,
    *,
    legal_mode: bool,
    user_input: str = "",
    jurisdiction: str = "PH",
    state=None,
) -> str:
    """Citation pipeline + ensure prefetched controlling authorities are named.

    Doctrine guards are jurisdiction-specific: the PH guards used to run on UK answers too
    (a UK defamation question could pick up a Civil Code Article 33 reminder)."""
    exhibits = exhibit_texts(state)  # the case's own documents: a legitimate source for quotations
    out = _finalize_legal_citations(
        text,
        search_results,
        legal_mode=legal_mode,
        extra_quote_sources=exhibits,
        # UK: only a blockquote formatted as a quotation is checked — proposed wording set as a
        # blockquote is not a claim about any source and must not be deleted as one.
        quotation_blocks_only=(jurisdiction == "UK"),
    )
    if legal_mode:
        out = append_missing_prefetched_mentions(out, search_results)
        if jurisdiction == "UK":
            out = append_uk_doctrine_guards(out, user_input, case_forum_text(state))
        elif jurisdiction == "PH":
            out = append_critical_doctrine_guards(out, user_input)
        # KR: no Korea-specific doctrine guards exist yet — neither the PH nor UK guard set
        # applies (a Korean-law answer picking up a Civil Code Art. 33 or PACE 1984 reminder
        # would be the same class of bug CONTEXT.md's "UK Doctrine Guards" note already fixed
        # for UK). Add Korean guards here if/when they're written.
        from legal_citations import (
            dedupe_repeated_citation_links as _dedupe_repeated_citation_links,
            format_legal_citation_links as _format_legal_citation_links,
            gate_unverified_legal_urls as _gate_unverified_legal_urls,
        )

        out = _gate_unverified_legal_urls(out, search_results)
        out = _format_legal_citation_links(out)
        if jurisdiction == "UK":
            # A closing list that re-links what the answer already cited reads as repetition.
            out = _dedupe_repeated_citation_links(out)
            # Quote marks claim the document's exact words: where the verifier round did not fix an
            # inline phrase that nearly matches an exhibit, drop the marks rather than misstate it.
            out = unquote_exhibit_misquotes(out, exhibits)
    out = _redact_document_ids(out, state)
    return out


def _is_intake_addendum(addendum_override) -> bool:
    """Detects the intake persona from its addendum text alone — the same idiom already used
    for `_legal_mode`/`_resume_persona` (string presence, not a passed-in persona id) — for the
    HITL-resume code paths (approve()/the WS 'approve' branch), which only ever have
    `addendum_override` (from state.pending_addendum), never the original `persona` value."""
    return bool(addendum_override) and "CASE INTAKE MODE" in addendum_override


def _finalize_intake_citations(text: str, search_results, state=None) -> str:
    """Citation gate for intake's general-rights answers — deliberately NOT
    _finalize_legal_response: no verifier/refine loop, no doctrine guards (those police
    *applied* legal argument; intake never makes one). Just the same repair → gate →
    format pipeline legal_uk uses, called directly as a pure function, so a citation intake
    names is either backed by a real tool result this turn or stripped — never invented.
    """
    exhibits = exhibit_texts(state)
    out = _finalize_legal_citations(
        text,
        search_results,
        legal_mode=True,
        extra_quote_sources=exhibits,
        quotation_blocks_only=True,
    )
    from legal_citations import dedupe_repeated_citation_links as _dedupe_repeated_citation_links

    out = _dedupe_repeated_citation_links(out)
    out = _redact_document_ids(out, state)
    return out
