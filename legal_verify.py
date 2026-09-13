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


# Phrases the CONTRADICTION SWEEP prompt rule (legal_prompt_uk.txt / legal_prompt.txt) asks the
# model to use when it resolves a contradiction ("state which side is more probative and why").
# A lexical heuristic, not a semantic check — it can't verify the resolution is actually
# correct, or credit a genuinely thorough answer that happens to phrase things differently. Kept
# deliberately generous (many phrasings) and paired with a modest threshold (see
# DEFAULT_MIN_CONTRADICTION_RESOLUTIONS) precisely because it's a proxy, not a ground truth —
# the goal is to catch a draft that plainly did no sweep at all, not to police wording.
_PROBATIVE_PHRASE = re.compile(
    r"\b(?:is more probative than|is (?:the )?more (?:probative|reliable|persuasive)|"
    r"is less (?:probative|reliable|persuasive)|is preferred (?:to|over)|"
    r"is to be preferred (?:to|over)|should be preferred (?:to|over)|outweighs|"
    r"carries (?:more|greater) weight than|takes precedence over|is the more reliable\b)",
    re.I,
)

DEFAULT_MIN_CONTRADICTION_RESOLUTIONS = 6


def count_contradiction_resolutions(text: str) -> int:
    """Heuristic count of resolved contradictions — see _PROBATIVE_PHRASE. Exported (rather
    than kept private) so it's directly unit-testable without going through AuditReport."""
    if not text:
        return 0
    return len(_PROBATIVE_PHRASE.findall(text))



# --- Authorities the user's own question names ------------------------------------------
# Precision over recall: every pattern here needs a strong "this is a case name" signal, because
# a false positive costs a full revise round on a real client turn. Recall gaps are acceptable —
# a name we fail to extract simply isn't checked, which is today's behaviour.
_AUTH_W = r"(?:R|[A-Z][\w'’-]+)"
_AUTH_SUFFIX = r"(?:plc|Ltd|LLP|Inc|Limited|Co)"
# One case-name side: up to three capitalised words, optionally one connector + one or two more
# words, optionally a corporate suffix. Bounded so "A v B and C v D" splits into two names.
_AUTH_NAME = (
    rf"{_AUTH_W}(?: {_AUTH_W}){{0,2}}(?: (?:and|&|de|of|le) {_AUTH_W}(?: {_AUTH_W})?)?"
    rf"(?: {_AUTH_SUFFIX})?"
)
_AUTH_V = re.compile(rf"\b({_AUTH_NAME}) v\.? ({_AUTH_NAME})\b")
_AUTH_LEADING_WORDS = {
    "in", "see", "apply", "applying", "consider", "considering", "address", "discuss", "does",
    "is", "was", "under", "per", "following", "cf", "compare", "contrast", "distinguish",
    "whether", "how", "why", "what", "did", "can", "should", "also", "and", "but", "the",
    "on", "for", "with", "from", "against", "citing", "given", "unlike", "like", "then",
}
_AUTH_PAREN_LIST = re.compile(r"\(([^()]{3,160})\)")
_AUTH_SLASH_LINE = re.compile(r"\b([A-Z][\w'’-]{2,}(?:/[A-Z][\w'’-]{2,})+)(?: line| principles| approach)?\b")
_AUTH_PURPOSES = re.compile(rf"\b({_AUTH_NAME}) purposes\b")
_AUTH_STOP = {
    "crown court", "employment tribunal", "high court", "court of appeal", "supreme court",
    "full code", "full code test", "england", "wales", "scotland", "northern ireland",
    "united kingdom", "part", "parts", "sch", "cpr", "pace", "hswa", "cja", "era", "gdpr", "dpa",
    "act", "regulations", "cps", "hse", "met office", "the", "and", "or", "of",
}


_AUTH_CONNECTORS = ("and", "&", "de", "of", "le", "v")


def _clean_authority(name: str) -> str:
    words = re.sub(r"\s+", " ", name).strip(" .,;:").split()
    while words and words[0].lower() in _AUTH_CONNECTORS:
        words.pop(0)
    while words and words[-1].lower() in _AUTH_CONNECTORS:
        words.pop()
    return " ".join(words).replace("&", "and")


def _looks_like_authority(name: str) -> bool:
    if not name or any(ch.isdigit() for ch in name):
        return False
    if name.lower() in _AUTH_STOP:
        return False
    words = name.split()
    if len(words) > 8:
        return False
    return all(w[0].isupper() or w in ("and", "de", "of", "le", "v", "plc", "the") for w in words)


def extract_named_authorities(query: str) -> List[str]:
    """Case names the user's message itself cites, in order of appearance, deduped.

    Catches `A v B`, `(Donoghue; Caparo)`-style parenthetical lists, `Hedley/Caparo`
    slash pairs and `… for Denton purposes`. Deliberately misses looser forms (see the
    note above the patterns).
    """
    if not query:
        return []
    found: List[str] = []

    def add(n: str) -> None:
        n = _clean_authority(n)
        if _looks_like_authority(n) and n.lower() not in {f.lower() for f in found}:
            found.append(n)

    pos = 0
    while True:
        m = _AUTH_V.search(query, pos)
        if not m:
            break
        left = m.group(1).split()
        if "R" in left:
            left = ["R"]
        while len(left) > 1 and left[0].lower() in _AUTH_LEADING_WORDS:
            left.pop(0)
        right = m.group(2)
        # "A v B and C v D": the connector clause of B is really the start of the next case
        # name whenever another " v " follows immediately — give it back and rescan from there.
        if " and " in right and re.match(r"\s+v\.?\s", query[m.end():]):
            right = right.rsplit(" and ", 1)[0]
            pos = m.start(2) + len(right)
        else:
            pos = m.end()
        add(f"{' '.join(left)} v {right}")
    for m in _AUTH_PAREN_LIST.finditer(query):
        inner = m.group(1)
        if ";" not in inner:
            continue
        cleaned = [_clean_authority(p) for p in inner.split(";")]
        if all(_looks_like_authority(c) for c in cleaned):
            for c in cleaned:
                add(c)
    for m in _AUTH_SLASH_LINE.finditer(query):
        for part in m.group(1).split("/"):
            add(part)
    for m in _AUTH_PURPOSES.finditer(query):
        add(m.group(1))
    return found


def _authority_tokens(name: str) -> List[str]:
    """Distinctive surname-like tokens a draft must contain to count as addressing `name`."""
    skip = {"and", "de", "of", "le", "v", "r", "the", "bank", "plc", "ltd", "limited", "co"}
    return [w for w in name.split() if w.lower() not in skip and len(w) >= 4]


def missing_named_authorities(draft: str, names: List[str]) -> List[str]:
    """Names from extract_named_authorities that the draft never mentions (case-insensitive;
    a name counts as mentioned when its distinctive token(s) appear as whole words)."""
    if not names:
        return []
    if not draft:
        return list(names)
    text = draft.replace("&", "and")

    def present(side: str) -> bool:
        if side == "R":  # the Crown — never distinctive
            return False
        # The first distinctive word is the short form lawyers actually use ("Caparo" for
        # "Caparo Industries plc"), so that alone establishes presence.
        token = (_authority_tokens(side) or [side])[0]
        return bool(re.search(rf"\b{re.escape(token)}\b", text, re.I))

    missing = []
    for name in names:
        # "Caparo Industries plc v Dickman" is addressed by "Caparo" alone — lawyers cite by the
        # short form of either party, so either side counts.
        sides = [p.strip() for p in re.split(r"\bv\b", name, maxsplit=1)]
        if not any(present(side) for side in sides if side):
            missing.append(name)
    return missing


@dataclass
class AuditReport:
    unverified_urls: List[Tuple[str, str]] = field(default_factory=list)
    truncated_urls: List[Tuple[str, str]] = field(default_factory=list)
    unverified_quotes: List[str] = field(default_factory=list)
    allowed: List[Tuple[str, str]] = field(default_factory=list)
    contradiction_resolutions: int = 0
    # 0 = the contradiction-sweep check is disabled for this turn (no case-document bundle to
    # sweep) — set by make_legal_verifier from state.case_document_manifest, never guessed here.
    min_contradiction_resolutions: int = 0
    # Authorities the user's question named that the draft never mentions (see
    # extract_named_authorities); empty when the question named none.
    missing_authorities: List[str] = field(default_factory=list)

    @property
    def contradiction_deficit(self) -> bool:
        return (
            self.min_contradiction_resolutions > 0
            and self.contradiction_resolutions < self.min_contradiction_resolutions
        )

    @property
    def has_issues(self) -> bool:
        return bool(
            self.unverified_urls
            or self.truncated_urls
            or self.unverified_quotes
            or self.contradiction_deficit
            or self.missing_authorities
        )

    @property
    def url_issue_count(self) -> int:
        return len(self.unverified_urls) + len(self.truncated_urls)

    def summary(self) -> str:
        parts = []
        if self.url_issue_count:
            parts.append(f"{self.url_issue_count} unverified citation link(s)")
        if self.unverified_quotes:
            parts.append(f"{len(self.unverified_quotes)} unverified quotation(s)")
        if self.contradiction_deficit:
            parts.append(f"only {self.contradiction_resolutions}/{self.min_contradiction_resolutions} contradictions resolved")
        if self.missing_authorities:
            parts.append(f"{len(self.missing_authorities)} user-named authority(ies) not addressed")
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


def audit_legal_draft(
    text: str,
    search_results,
    *,
    min_contradiction_resolutions: int = 0,
    named_authorities: Optional[List[str]] = None,
) -> AuditReport:
    """Report (never rewrite) every citation link / blockquote the finalizer would demote, plus
    (when min_contradiction_resolutions > 0 — see AuditReport) whether the draft shows enough
    resolved contradictions to look like a real sweep happened."""
    report = AuditReport(
        allowed=collect_tool_result_citables(search_results),
        min_contradiction_resolutions=min_contradiction_resolutions,
    )
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

    report.contradiction_resolutions = count_contradiction_resolutions(text)
    report.missing_authorities = missing_named_authorities(text, named_authorities or [])
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

    if report.contradiction_deficit:
        lines.append("")
        lines.append(
            f"Contradiction sweep: the draft states a probative resolution (\"X is more "
            f"probative/reliable than Y, because...\") for only about "
            f"{report.contradiction_resolutions} contradiction(s). A case-document set with "
            f"multiple exhibits and witness accounts usually has considerably more than that. "
            f"Re-scan by fact-type — dates and durations, quantities, presence/absence claims, "
            f"authorship/authenticity, and a document's own internal arithmetic — checking "
            f"every source against every OTHER source that speaks to the same fact, not just "
            f"the pair that's most obvious. For each new contradiction you find, add one "
            f"sentence stating which side is more probative and why."
        )

    if report.missing_authorities:
        lines.append("")
        lines.append("Authorities the user's question itself names but the draft never mentions:")
        for name in report.missing_authorities:
            lines.append(f"- {name}")
        lines.append(
            "A user-named authority must be addressed BY NAME — never silently dropped because "
            "it could not be hyperlinked. For each: (a) search for it (PH: search_jurisprudence; "
            "UK: case_law_search with the case name) and, if the tool returns the judgment or a "
            "modern judgment that discusses it, read the relevant paragraph and state its "
            "proposition from that retrieved text; (b) if no retrievable source discusses it, "
            "still name it in plain text WITHOUT a hyperlink, say that it could not be "
            "retrieved, and confine yourself to how it bears on the question; (c) if it is not "
            "in point, say expressly why it is distinguished. Never fabricate a URL, neutral "
            "citation, or holding for it."
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


def make_legal_verifier(state, *, max_rounds: Optional[int] = None, query: str = "") -> Optional[Verifier]:
    """Closure the chain loops call with (draft_text, rounds_done) when the model
    stops calling tools. Returns feedback text to inject, or None to accept the
    draft. Reads state.last_search_legal_results at call time because the pool
    grows during the turn (a revise round may retrieve more authorities)."""
    rounds = verify_max_rounds_from_env() if max_rounds is None else max_rounds
    if rounds <= 0:
        return None
    named = extract_named_authorities(query)

    def verify(draft: str, rounds_done: int) -> Optional[VerifierFeedback]:
        from the_server import increment_metric_counter

        # Only meaningful when there's an actual case-document bundle to sweep for cross-
        # document contradictions — a plain legal question with no attached exhibits has
        # nothing to sweep, and demanding the phrasing anyway would just force pointless
        # revise rounds on unrelated turns.
        manifest_len = len(getattr(state, "case_document_manifest", None) or [])
        min_contradictions = DEFAULT_MIN_CONTRADICTION_RESOLUTIONS if manifest_len >= 2 else 0
        report = audit_legal_draft(
            draft,
            getattr(state, "last_search_legal_results", None),
            min_contradiction_resolutions=min_contradictions,
            named_authorities=named,
        )
        tags = {
            "round": str(rounds_done),
            "urls": str(report.url_issue_count),
            "quotes": str(len(report.unverified_quotes)),
            "contradictions": str(report.contradiction_resolutions),
            "missing_authorities": str(len(report.missing_authorities)),
        }
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
