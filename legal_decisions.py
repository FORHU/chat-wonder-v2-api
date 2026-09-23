# -*- coding: utf-8 -*-
"""Decision Records — SCL auditability beyond hyperlinks (differentiation program, Phase 1).

A hyperlink says "this sentence came from this source." A decision record says why: what rule
was applied, what evidence supported the conclusion, what evidence cut against it, what
alternative reading was considered and why it was rejected, how the evidence was weighted, and
what fact would change the answer. This module is the pure, testable half of that: given the
model's own draft records, it verifies every authority link and every evidence reference against
what was ACTUALLY retrieved this turn — the same discipline legal_citations/legal_verify already
apply to the answer's own citations, extended from sentences to reasons.

The record schema (as the model returns it, before audit):
    {
      "anchor": str,        # a verbatim sentence copied from the final answer
      "conclusion": str,    # the proposition the anchor asserts, in one sentence
      "rule": [{"title": str, "url": str}],                    # authority relied on
      "evidenceFor": [{"doc": str, "pinpoint": str, "quote": str}],      # doc = handle, e.g. "F1"
      "evidenceAgainst": [{"doc": str, "pinpoint": str, "quote": str}],
      "alternatives": [{"position": str, "whyRejected": str, "evidenceRef": str}],
      "weighting": str,
      "confidence": "high" | "medium" | "low",
      "wouldChangeIf": [str],
    }

After audit_decision_records: `rule` entries whose url doesn't resolve have url=None,
verified=False (kept as plain text, never shipped as a clickable link — same posture as the
Cite Gate); `evidenceFor`/`evidenceAgainst` entries get `docId` (resolved case-document id, or
None), `doc` rewritten to the exhibit's real file name (never the model's raw handle/id — see
"Fix: the file ID shows instead of the file name"), and `verified` (docId resolved AND, if a
quote was given, the quote is actually in that document's/the retrieved pool's text). Nothing is
silently dropped from evidence — an unverifiable item is informative (it shows what the model
could not itself substantiate) — but `rule` links are dropped to a bare title, exactly like the
Cite Gate does for the answer itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from legal_citations import (
    _normalize_quote_text,
    allowed_url_set,
    classify_href,
    collect_tool_result_text_corpus,
    quote_appears_in_corpus,
)

_MAX_RECORDS = 8
_DOC_LABEL_PREFIX_RE = re.compile(r"^([A-Za-z]{1,4}\d{1,4}(?:\.\d+)?)\b")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.IGNORECASE)


def _looks_like_uuid(text: str) -> bool:
    return bool(_UUID_RE.match((text or "").strip()))


@dataclass
class DecisionAuditStats:
    records_in: int = 0
    records_out: int = 0
    anchors_unverified: int = 0
    rules_verified: int = 0
    rules_dropped: int = 0
    evidence_verified: int = 0
    evidence_unverified: int = 0
    quotes_verified: int = 0
    quotes_unverified: int = 0

    def as_tags(self) -> Dict[str, str]:
        return {k: str(v) for k, v in self.__dict__.items()}


def case_document_text_corpus(case_documents: Optional[List[dict]]) -> str:
    """Flatten the full text of every case document currently held in context — the
    case-document counterpart of legal_citations.collect_tool_result_text_corpus, which only
    walks the public jurisprudence/legislation pool. Used so an evidence quote can be verified
    against the user's own exhibits, not only against retrieved authorities."""
    if not case_documents:
        return ""
    parts = [d.get("text") or "" for d in case_documents if isinstance(d, dict)]
    return _normalize_quote_text(" ".join(p for p in parts if p))


def case_document_labels(
    case_documents: Optional[List[dict]],
    manifest: Optional[List[dict]],
    handles: Optional[Dict[str, str]] = None,
) -> List[Tuple[str, str]]:
    """(label, id) pairs an evidence reference's `doc` field might match: the document's
    per-session handle (e.g. "F1", see the_server.py's doc_handle_by_id — what the model is
    actually shown in its prompts now), every attached exhibit's manifest name in full, plus —
    for the "D07_Interview_..." bundle-document naming convention this product's benchmarks
    use — the short code alone ("D07"), so a model that writes `"doc": "D07"` resolves to the
    same document as one that writes the full filename. The raw id itself is also matched as a
    defensive fallback (an older session, or a model that ignores the handle instruction) — it
    is never what's shown back to a user; resolve_doc_label's exact-match-first pass means a
    handle like "F1" is never mistaken for a manifest name that happens to start with "F1".
    Falls back to case_documents (name/id) when no manifest was supplied."""
    pairs: List[Tuple[str, str]] = []
    source = manifest if manifest else (case_documents or [])
    for m in source:
        if not isinstance(m, dict):
            continue
        doc_id = m.get("id")
        name = m.get("name") or ""
        if not doc_id or not name:
            continue
        if handles and str(doc_id) in handles:
            pairs.append((handles[str(doc_id)], doc_id))
        pairs.append((name, doc_id))
        short = _DOC_LABEL_PREFIX_RE.match(name)
        if short:
            pairs.append((short.group(1), doc_id))
        pairs.append((str(doc_id), doc_id))
    return pairs


def resolve_doc_label(label: str, labels: List[Tuple[str, str]]) -> Optional[str]:
    """Best-effort match of an evidence `doc` field to an attached exhibit's id. Exact
    (case-insensitive) match first, then a prefix/substring match against the full manifest
    name — "D01" against "D01_HSE_Preliminary_Investigation_Report.pdf", or vice versa."""
    if not label:
        return None
    norm = label.strip().lower().rstrip(".:,;")
    if not norm:
        return None
    for name, doc_id in labels:
        if norm == name.lower():
            return doc_id
    for name, doc_id in labels:
        n = name.lower()
        if n.startswith(norm) or norm.startswith(n) or norm in n:
            return doc_id
    return None


def _normalize_anchor(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip()


def _audit_evidence(
    items: Any,
    labels: List[Tuple[str, str]],
    corpus: str,
    stats: DecisionAuditStats,
    names: Optional[Dict[str, str]] = None,
) -> List[dict]:
    names = names or {}
    audited: List[dict] = []
    for e in items or []:
        if not isinstance(e, dict):
            continue
        doc_label = str(e.get("doc") or "").strip()
        doc_id = resolve_doc_label(doc_label, labels)
        quote = e.get("quote")
        quote_ok = True
        if quote:
            quote_ok = quote_appears_in_corpus(str(quote), corpus)
            if quote_ok:
                stats.quotes_verified += 1
            else:
                stats.quotes_unverified += 1
        verified = bool(doc_id) and quote_ok
        if doc_id:
            stats.evidence_verified += 1
        else:
            stats.evidence_unverified += 1
        # `doc` is what the UI shows as the source — always the resolved file name, never the
        # model's raw label (a handle, or a leaked UUID). A UUID-shaped label that failed to
        # resolve is masked to "Document" rather than shown raw (see the file-ID-shown-instead-
        # of-name fix); a non-UUID label that failed to resolve (a genuine typo) is kept as-is
        # so it's still informative for debugging.
        display_doc = names.get(str(doc_id)) if doc_id else None
        if not display_doc:
            display_doc = "Document" if _looks_like_uuid(doc_label) else doc_label
        audited.append(
            {
                "doc": display_doc,
                "docId": doc_id,
                "pinpoint": e.get("pinpoint") or "",
                "quote": quote or None,
                "verified": verified,
            }
        )
    return audited


def audit_decision_records(
    records: Any,
    legal_response: str,
    search_results,
    case_documents: Optional[List[dict]],
    manifest: Optional[List[dict]],
    handles: Optional[Dict[str, str]] = None,
) -> Tuple[List[dict], DecisionAuditStats]:
    """Verify a batch of model-produced decision records:
      - `anchor` must be a verbatim (whitespace-normalized) substring of the actual answer —
        a record that doesn't anchor to real text in the response is dropped outright, the same
        way RedTeamAssessment/CaseReconstruction claims are matched by exact substring at render
        time (see attributed-text.tsx) rather than trusted as free-standing text.
      - every `rule[].url` must be an exact URL from this turn's retrieved pool (classify_href,
        same rule as the Cite Gate); an unverified or truncated url is dropped to a bare title.
      - every `evidenceFor`/`evidenceAgainst` item's `doc` must resolve to an attached exhibit;
        a `quote`, if given, must appear in the retrieved/case-document text corpus.
    Returns (audited_records, stats) — stats are logged as metrics by the caller.
    """
    stats = DecisionAuditStats()
    if not isinstance(records, list):
        return [], stats
    stats.records_in = len(records)

    normalized_answer = _normalize_anchor(legal_response)
    allowed_urls = allowed_url_set(search_results)
    corpus = " ".join(
        c for c in (collect_tool_result_text_corpus(search_results), case_document_text_corpus(case_documents)) if c
    )
    labels = case_document_labels(case_documents, manifest, handles)
    names = {
        str(m.get("id")): m.get("name")
        for m in (manifest if manifest else (case_documents or []))
        if isinstance(m, dict) and m.get("id") and m.get("name")
    }

    out: List[dict] = []
    for rec in records[:_MAX_RECORDS]:
        if not isinstance(rec, dict):
            continue
        anchor = str(rec.get("anchor") or "")
        if not anchor or _normalize_anchor(anchor) not in normalized_answer:
            stats.anchors_unverified += 1
            continue

        rule_out = []
        for r in rec.get("rule") or []:
            if not isinstance(r, dict):
                continue
            title = str(r.get("title") or "").strip()
            if not title:
                continue
            url = str(r.get("url") or "").strip()
            verdict = classify_href(url, allowed_urls) if url else "skip"
            if url and verdict == "ok":
                rule_out.append({"title": title, "url": url, "verified": True})
                stats.rules_verified += 1
            else:
                rule_out.append({"title": title, "url": None, "verified": False})
                stats.rules_dropped += 1

        out.append(
            {
                "anchor": anchor,
                "conclusion": str(rec.get("conclusion") or "").strip(),
                "rule": rule_out,
                "evidenceFor": _audit_evidence(rec.get("evidenceFor"), labels, corpus, stats, names),
                "evidenceAgainst": _audit_evidence(rec.get("evidenceAgainst"), labels, corpus, stats, names),
                "alternatives": [
                    {
                        "position": str(a.get("position") or "").strip(),
                        "whyRejected": str(a.get("whyRejected") or "").strip(),
                        "evidenceRef": str(a.get("evidenceRef") or "").strip() or None,
                    }
                    for a in (rec.get("alternatives") or [])
                    if isinstance(a, dict) and (a.get("position") or a.get("whyRejected"))
                ],
                "weighting": str(rec.get("weighting") or "").strip(),
                "confidence": rec.get("confidence") if rec.get("confidence") in ("high", "medium", "low") else "medium",
                "wouldChangeIf": [str(w).strip() for w in (rec.get("wouldChangeIf") or []) if str(w).strip()],
            }
        )
    stats.records_out = len(out)
    return out, stats


DECISION_RECORDS_SCHEMA_PROMPT = """Return ONLY a JSON object: {"records": [...]}.

Produce 3-6 decision records — one per CONSEQUENTIAL or CONTESTED conclusion in the answer below
(not routine/uncontroversial statements). Each record:
{
  "anchor": string,            REQUIRED. Copy one sentence VERBATIM, character-for-character,
                                from the "FINAL ANSWER" below (including its punctuation). This
                                is used to locate the sentence in the text — if it does not match
                                exactly, the record is discarded, so copy, do not paraphrase.
  "conclusion": string,        the proposition that sentence asserts, in your own words
  "rule": [{"title": string, "url": string}],   the authority (case/statute) it rests on, using
                                the EXACT resolved url already used in the answer for that
                                authority — never a new or guessed url
  "evidenceFor": [{"doc": string, "pinpoint": string, "quote": string}],
                                doc = the exhibit's handle as it appears in the case file (e.g.
                                "F1") — never its id; pinpoint = paragraph/part/item; quote = a
                                short phrase copied VERBATIM from that document's SOURCE
                                EXCERPTS below, character-for-character (omit quote if you are
                                paraphrasing, or if no SOURCE EXCERPTS were given for that
                                document — a sentence from your own answer is not evidence)
  "evidenceAgainst": [...],    same shape — evidence that cuts against the conclusion, if any
  "alternatives": [{"position": string, "whyRejected": string, "evidenceRef": string}],
                                a different reading of the same facts/law that was considered and
                                rejected, and why — omit only if there is genuinely none
  "weighting": string,         one sentence on how the for/against evidence was weighed
  "confidence": "high" | "medium" | "low",
  "wouldChangeIf": [string]    1-3 concrete facts that, if different, would change this
                                conclusion
}

Do not invent a rule, doc, pinpoint, or quote that isn't already in the answer or the sources
retrieved this turn. If a conclusion has no real alternative or no authority beyond the case
facts, omit that field's items rather than inventing them."""
