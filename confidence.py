# -*- coding: utf-8 -*-
"""Signal-agnostic confidence-span layer.

Thermo Chat colors text by per-token Shannon entropy — a purely statistical signal with no idea
where a claim came from. This module measures something different and, for a legal product, more
useful: PROVENANCE. For every sentence of the delivered answer, is it (1) quoted or very closely
paraphrased from an identified source, (2) the model's own conclusion but drawing on an identified
source, or (3) the model's own reasoning with no source behind it at all? Three categories, one
formula, no LLM judge call — see provenance_spans' docstring for the exact equations, specified
with the same rigor as Thermo Chat's own H_t/T_t (named constants, stated bounds, no hidden
tuning).

This is module history worth keeping, because the three earlier designs it replaced are each a
real, documented lesson:
  1. attribution_spans / citation_spans (citation-link and attribution-check reuse): only ever
     covered a sentence that happened to cite a source or attribute a statement to a document.
     Most answers, and every answer that cited nothing, got NOTHING from either — the opposite
     of what "work on reasoning models... luna and terra" requires.
  2. self_consistency_spans (K-sample resampling + one LLM judge call): fixed the coverage gap
     by sampling the same turn multiple times and asking a judge where the samples agreed,
     contradicted, or stayed silent. Real signal, but expensive (K extra generations + a judge
     call per turn, live-tested at roughly 4-5x cost) and an indirect proxy for the actual
     question a reader asks, which is provenance, not cross-sample agreement.
  3. provenance_spans (current): answers the provenance question directly, and with a formula
     instead of a second judge call — zero extra LLM cost, deterministic, testable without a
     fake judge. The tradeoff, stated plainly: lexical overlap is not semantic understanding. It
     can be fooled by coincidental word overlap, and it will miss a paraphrase that reuses none
     of a source's actual vocabulary. See provenance_spans' docstring for the formula and its
     stated, named constants — exactly the kind of thing that needs calibrating against real
     traffic before the thresholds below should be trusted (same posture the architecture doc's
     "Calibration" section already takes toward the entropy display bands).

Also still here: real per-token Shannon entropy from Chat Completions logprobs, for the dev
testers/staff only (gpt-4o*, gpt-4.1*, deepseek-chat -- see model_registry.supports_logprobs).

Reference:
  Kim, M. H. (2026). "Thermo Pulse: Entropy-Feedback Temperature Control for Autoregressive
  Language Model Decoding." Preprints.org, posted 4 November 2026.
  https://doi.org/10.20944/preprints202611.0277.v1 -- licensed CC BY 4.0.
token_entropy reimplements that paper's Eq. 1 (renormalized top-5 entropy, bounded by ln 5);
group_words follows its word-level display rule (a word's value is the MAX over its tokens). The
Thermo Chat demo package itself is All Rights Reserved; nothing here is copied from it.

What we can and cannot reproduce on OpenAI (each verified live, 2026-11-10):
  - CAN: H_t. top_logprobs=5 gives the five highest log-probs, which are pre-temperature (they
    barely move between temperature 0 and 1.8), i.e. the raw-logit distribution Eq. 1 requires.
    token_entropy matches an independent Eq. 1 implementation to ~5e-12 on live gpt-4o-mini
    tokens; mean H was 0.33 nats, in line with the preprint's 0.26-0.29.
  - CANNOT: the Thermo Pulse controller (Eq. 2, per-token temperature T_t) or its 5% cutoff.
    The API takes one fixed temperature per request (a per-token value is rejected outright),
    so there is no T_t here -- publishing a "temperature" we never applied would be
    fabrication. Only h (nats) and hn = h/ln5 (0-1) are emitted.
  - CANNOT on reasoning models: gpt-5.6-luna/terra reject logprobs on both APIs ("logprobs are
    not supported with reasoning models"), so production legal answers get no H_t at all --
    the reason provenance_spans exists as the signal that does work on them.
One deliberate divergence from the demo package: its remote/vLLM path computes entropy over the
raw top-5 probabilities WITHOUT renormalizing, a different quantity from Eq. 1. We renormalize,
as the paper and the package's own local engine do.

Fails open throughout: any error here must never alter or block an answer, only omit a span.
"""

from __future__ import annotations

import html
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

SCHEMA_VERSION = 1
LN5 = math.log(5)


def confidence_enabled() -> bool:
    """CONFIDENCE_SPANS: ON by default, following the repo's own default-on idiom
    (LEGAL_USE_RESPONSES_API at the_server.py:2880, LEGAL_ATTRIBUTION_CHECK at
    attribution_check.py:104). Read fresh on every call so it is flippable without a restart.
    This is a behavior flag, not a secret: it must not be added to the committed .env."""
    return os.getenv("CONFIDENCE_SPANS", "true").strip().lower() not in ("0", "false", "no", "off")


@dataclass
class Span:
    quote: str
    tier: str  # "QUOTED" | "SOURCED" | "UNSOURCED"
    source: str  # "citation" | "provenance" | "entropy"
    why: str = ""
    ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "quote": self.quote,
            "tier": self.tier,
            "source": self.source,
            "why": self.why,
            "ref": self.ref,
        }


# ---------------------------------------------------------------------------
# Entropy (Chat Completions logprobs only -- see module docstring)
# ---------------------------------------------------------------------------

def token_entropy(logprobs: List[Optional[float]]) -> Optional[float]:
    """Renormalized top-5 Shannon entropy in nats (preprint Eq. 1: H_t = -sum(q_i * ln q_i),
    q renormalized so sum(q) = 1, bounded by ln(5) ~= 1.6094). `logprobs` is the list of raw
    logprob floats from one token's top_logprobs. Returns None if nothing usable is given --
    callers must treat None as "no signal", never as 0."""
    probs = [
        math.exp(min(lp, 0.0))
        for lp in (logprobs or [])
        if isinstance(lp, (int, float)) and math.isfinite(lp)
    ]
    if not probs:
        return None
    total = sum(probs)
    if total <= 0:
        return None
    q = [p / total for p in probs]
    return -sum(p * math.log(p + 1e-12) for p in q)


def group_words(entries: List[tuple]) -> List[dict]:
    """entries: [(token_text, h_or_None)] in emission order, built from logprobs.content[].token
    -- never from the yielded delta text, which the_server.py's streaming loop runs through
    .replace("~","-") and first-line REASON: buffering before it reaches the client. A new word
    starts whenever a token's text begins with whitespace/newline (matches the Thermo Chat
    reference's own client-side rule, and the preprint's stated word-level display). A word's
    score is the MAX h among its own tokens that have a measured h -- matching the reference
    exactly (its README: "the interface uses the maximum entropy ... observed among its measured
    tokens")."""
    words: List[dict] = []
    cur_text = ""
    cur_hs: List[Optional[float]] = []

    def flush():
        text = cur_text.strip()
        if not text:
            return
        measured = [h for h in cur_hs if h is not None]
        h = max(measured) if measured else None
        words.append({
            "w": text,
            "h": round(h, 4) if h is not None else None,
            "hn": round(h / LN5, 4) if h is not None else None,
            "n": len(cur_hs),
        })

    for token_text, h in entries:
        token_text = token_text or ""
        is_new_word = bool(token_text) and token_text[0] in (" ", "\n", "\t") and cur_text
        if is_new_word:
            flush()
            cur_text = ""
            cur_hs = []
        cur_text += token_text
        cur_hs.append(h)
    flush()
    return words


def entropy_payload(entries: List[tuple]) -> Optional[dict]:
    if not entries:
        return None
    words = group_words(entries)
    if not words:
        return None
    return {
        "unit": "nats",
        "ceiling": round(LN5, 4),
        "metric_source": "top5_renormalized",
        "words": words,
    }


def to_live_frame(token_text: str, h: Optional[float]) -> str:
    """One [CONF]...[/CONF] frame per token, sent only when ChatRequest.confidence_live is set
    (dev testers; the relay never sends this field -- see model_registry.supports_logprobs and
    ChatRequest.confidence_live). Mirrors the [TRACE] sentinel pattern exactly: the websocket
    handler must send_text() this and then `continue`, the same way it already does for [TRACE],
    so it never reaches full_response. That `continue` is the whole safety property the previous
    attempt's [METRIC] frames omitted."""
    payload = {
        "token": token_text,
        "h": round(h, 4) if h is not None else None,
        "hn": round(h / LN5, 4) if h is not None else None,
    }
    return "[CONF]" + json.dumps(payload, ensure_ascii=False) + "[/CONF]"


# ---------------------------------------------------------------------------
# Sentence extraction -- matches what marked.parse() actually renders
# ---------------------------------------------------------------------------
#
# A span's quote is matched CLIENT-SIDE against the text of the RENDERED markdown DOM, not
# final_text's raw markdown source (see confidence-render.js's render(): it always renders
# markdown first, via marked.parse(), then overlays spans by walking the result's text nodes).
# Markdown rendering strips "#"/"-"/">"/"**" markup and un-wraps "[label](url)" links to their
# label text, so a quote has to be built the SAME way -- cleaned, not verbatim -- or it will
# never be found in the DOM marked.parse() actually produces. _plain_for_match does exactly the
# same cleanup marked.js performs, so the two line up.

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9])")
_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
_TABLE_SEP_CELL = re.compile(r"^:?-{1,}:?$")
_WORD_RE = re.compile(r"[a-z0-9]+(?:['’][a-z]+)?")

_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it of on or that the their this to was were "
    "which with should would could may can not no must also any each both same into than then there "
    "these those its his her him she he they them said says say do does did done be been being "
    "will shall if but so such more most other some such no nor only own same so than too very just "
    "about above after again against all am because before below between both down during "
    "further here how once under until while who whom why".split()
)


#: Markdown link whose label may contain ONE level of nested brackets -- UK neutral citations
#: put "[2010]" inside the label ("[Gold Group v BDW [2010] EWHC 323 Jurisprudence](url)"),
#: which a plain [^\]]+ label pattern stops at, leaving raw markdown in the quote (found live).
_MD_LINK_RE = re.compile(r"\[((?:[^\[\]]|\[[^\[\]]*\])+)\]\(([^)\s]+)\)")
_MD_HTTP_LINK_RE = re.compile(r"\[((?:[^\[\]]|\[[^\[\]]*\])+)\]\((https?://[^)\s]+)\)")


def _plain_for_match(s: str) -> str:
    """Cleans one line/sentence down to what marked.parse() will actually render as visible
    text, so a span's quote matches a real text node instead of silently never matching
    markdown-syntax characters that never reach the DOM."""
    s = re.sub(r"<[^>]+>", "", s)
    s = _MD_LINK_RE.sub(r"\1", s)
    s = re.sub(r"^#{1,6}\s*", "", s)
    s = re.sub(r"^[-*+]\s+", "", s)
    # Ordered-list markers too: marked renders "1. " as <ol> numbering, never as text, so a
    # quote that kept it could never match the DOM (found live: spans silently dropped).
    s = re.sub(r"^\d{1,3}[.)]\s+", "", s)
    s = re.sub(r"^>\s?", "", s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\1", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", s)
    s = re.sub(r"`([^`]+)`", r"\1", s)
    return re.sub(r"\s+", " ", s).strip()


#: Safety bound only -- NOT a coverage choice. An earlier hard limit of 80 left everything after
#: the first screenful of a long answer unshaded (found live: 54 of 407 blocks covered on a
#: 31K-char UK answer). Real answers stay far below this; it exists so a pathological output
#: can't produce an unbounded payload.
MAX_CANDIDATE_SENTENCES = 2000


def _candidate_sentences(text: str, limit: int = MAX_CANDIDATE_SENTENCES) -> List[str]:
    """Cleaned sentence/cell text, matching what the rendered DOM will contain -- NOT verbatim
    substrings of final_text's raw markdown source (see the module-level comment above for
    why). Nothing is skipped by line type: headings, bullets and blockquotes are all candidates,
    because leaving any of them uncolored reads as "this part was never checked" next to
    sentences that were -- the opposite of what coverage should communicate.

    Two things ARE still skipped, for a structural reason rather than a content one:
      - Any line containing an embedded citation anchor (`<a href=`): citation_spans already
        covers that link precisely via href matching, and a sentence spanning INTO an anchor
        element can never be found as one contiguous text node after rendering (the anchor's
        text lives in its own child text node, split off from the surrounding prose) -- so
        matching it here would silently never work anyway.
      - A markdown table row is split into its individual CELLS instead of treated as one
        sentence, for the identical structural reason: marked.js renders each cell as its own
        <td> with its own isolated text node, so only a per-cell quote can ever match."""
    out: List[str] = []
    for line in (text or "").splitlines():
        stripped = line.lstrip()
        if not stripped or "<a href=" in stripped:
            continue
        # Markdown http(s) links get the same treatment as embedded <a href> anchors: the link
        # is left to quoted_citation_spans (href-matched, styled in place), so it's removed
        # here rather than also wrapped in a <mark> inside its own <a>. Unlike the HTML case,
        # the surrounding prose on the line stays a candidate -- marked renders the link as a
        # separate inline element, so the prose before/after it is still matchable.
        stripped = _MD_HTTP_LINK_RE.sub("", stripped).strip()
        if not stripped:
            continue
        table_row = _TABLE_ROW.match(stripped)
        if table_row:
            for cell in table_row.group(1).split("|"):
                cleaned = _plain_for_match(cell)
                if cleaned and not _TABLE_SEP_CELL.match(cleaned):
                    out.append(cleaned)
                    if len(out) >= limit:
                        return out
            continue
        for raw_sentence in _SENT_SPLIT.split(stripped):
            cleaned = _plain_for_match(raw_sentence)
            if len(cleaned) >= 3:
                out.append(cleaned)
                if len(out) >= limit:
                    return out
    return out


#: Same character class legal_citations._normalize_quote_text strips. Sources built from the
#: legal pool come through collect_tool_result_text_corpus, which already removed these, so the
#: sentence side has to drop them too or a real quotation (“...”) can never verbatim-match.
_QUOTE_CHARS = re.compile(r'[\"“”‘’*_`]')


_MARKUP_TAG_RE = re.compile(r"<[^<>]{0,2000}>")


def _clean_source(text: str) -> str:
    """Source text as a reader would see it. National Archives judgments arrive as Akoma Ntoso
    XML with HTML entities (found live: Lord Radcliffe's dictum stored as
    '<p class=quote>&#8220;&#8230;frustration occurs ...'), so a verbatim quotation of it
    never matched, and markup words ("xmlns", "oasis", "akn") leaked into the vocabulary.
    Entities are decoded and tags dropped before any matching or scoring."""
    if not text:
        return ""
    text = html.unescape(_MARKUP_TAG_RE.sub(" ", html.unescape(str(text))))
    return re.sub(r"\s+", " ", text).strip()


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", _QUOTE_CHARS.sub("", text or "")).strip().casefold()


def _toks(text: str) -> Set[str]:
    """Casefolded content words, stopwords and short tokens (<3 chars) dropped -- distinctive
    vocabulary only, the same idea attribution_check.py's own tokenizer uses for its excerpt
    relevance scoring."""
    return {w for w in _WORD_RE.findall((text or "").casefold()) if len(w) >= 3 and w not in _STOPWORDS}


# ---------------------------------------------------------------------------
# Provenance spans -- the formula
# ---------------------------------------------------------------------------

#: QUOTED requires a near-verbatim match at least this many characters long, so two short,
#: generic phrases ("the employer") that happen to coincide don't register as a quotation.
QUOTE_MIN_CHARS = 20

#: SOURCED requires the IDF-weighted coverage score C(S) (see provenance_spans' docstring for
#: the formula) to reach this fraction. STARTING VALUE, stated plainly like Thermo Chat's own
#: preprint states its alpha/beta/H* constants: "fixed without a separate tuning set... a
#: systematic search is left for future work." Calibrate against real traffic before trusting
#: this number -- see the architecture doc's "Calibration" section, which already takes this
#: posture toward the entropy display bands and should take the same one here.
SOURCED_COVERAGE_THRESHOLD = 0.45


def _is_verbatim_match(sentence: str, sources: List[str], *, normalized: bool = False) -> bool:
    """Q(S) in the module docstring's formula: 1 if the normalized sentence (or a long enough
    normalized source passage) is a substring of the other, else 0. QUOTE_MIN_CHARS keeps a
    short, generic sentence from registering as a "quotation" of an unrelated source purely by
    coincidence.

    `normalized=True` means `sources` were already passed through _normalize -- provenance_spans
    does that once per turn, since sources are now full statute/judgment texts and
    re-normalizing every one of them for every sentence would be quadratic in answer length."""
    norm_s = _normalize(sentence)
    if len(norm_s) < QUOTE_MIN_CHARS:
        return False
    # An elided quotation ("…frustration occurs … radically different …") is QUOTED when every
    # substantive fragment appears in ONE source -- never stitched across different sources.
    # Fragments shorter than QUOTE_MIN_CHARS are connective ("and", "a thing") and are ignored,
    # but at least one fragment must clear that bar.
    fragments = [f.strip(" ,;:.") for f in _ELLIPSIS_RE.split(norm_s)]
    fragments = [f for f in fragments if len(f) >= QUOTE_MIN_CHARS]
    if not fragments:
        return False
    elided = len(fragments) > 1 or _ELLIPSIS_RE.search(norm_s) is not None
    for src in sources:
        norm_src = src if normalized else _normalize(src)
        if not norm_src:
            continue
        if norm_s in norm_src:
            return True
        if elided and all(f in norm_src for f in fragments):
            return True
        if len(norm_src) >= QUOTE_MIN_CHARS and norm_src in norm_s:
            return True
        if _long_verbatim_run(norm_s, norm_src):
            return True
    return False


#: A long sentence is also QUOTED when one contiguous verbatim run covers at least this share of
#: it AND at least QUOTE_RUN_MIN_CHARS -- for a quotation whose only retrieved copy is a snippet
#: cut mid-sentence (found live: "eing performed because ..."). 100 identical characters in a row
#: is far beyond coincidental overlap. Uncalibrated, like the other constants here.
QUOTE_RUN_MIN_SHARE = 0.5
QUOTE_RUN_MIN_CHARS = 100
_RUN_WINDOW, _RUN_STEP = 40, 8


def _long_verbatim_run(norm_s: str, norm_src: str) -> bool:
    """Longest run of consecutive 40-char windows of the sentence that all occur in the source,
    each found right after the previous one (so the run is contiguous in the source too)."""
    need = max(QUOTE_RUN_MIN_CHARS, int(QUOTE_RUN_MIN_SHARE * len(norm_s)))
    if len(norm_s) < need or len(norm_src) < need:
        return False
    best = 0
    i = 0
    while i + _RUN_WINDOW <= len(norm_s):
        pos = norm_src.find(norm_s[i:i + _RUN_WINDOW])
        if pos == -1:
            i += _RUN_STEP
            continue
        start, j = i, i
        while j + _RUN_STEP + _RUN_WINDOW <= len(norm_s) and \
                norm_src.startswith(norm_s[j + _RUN_STEP:j + _RUN_STEP + _RUN_WINDOW], pos + (j + _RUN_STEP - start)):
            j += _RUN_STEP
        # Extend to the true end of the matching stretch (up to one step beyond the last window).
        end = j + _RUN_WINDOW
        while end < len(norm_s) and pos + (end - start) < len(norm_src) and norm_s[end] == norm_src[pos + (end - start)]:
            end += 1
        best = max(best, end - start)
        if best >= need:
            return True
        i = j + _RUN_STEP
    return False


_ELLIPSIS_RE = re.compile(r"\s*(?:…|\.\.\.|\[\.\.\.\]|\[…\])\s*")


def _coverage_score(sentence_toks: Set[str], idf: Dict[str, float]) -> float:
    """C(S) in the module docstring's formula: the IDF-weighted fraction of the sentence's own
    distinctive vocabulary that also appears somewhere in the source corpus.

        C(S) = sum(idf(w) for w in T(S) if w in corpus vocabulary) / sum(idf(w) for w in T(S))

    where T(S) is the sentence's distinctive (non-stopword) word set. A word idf() has no entry
    for (meaning it never appeared in the corpus at all) contributes 0 to the numerator and
    idf.get(w, 1.0) to the denominator -- an unseen word still counts against coverage with a
    neutral weight, rather than being silently dropped from the computation, which would let a
    sentence made entirely of corpus-absent words score an undefined or inflated ratio. Returns
    0.0 for a sentence with no distinctive vocabulary at all (nothing to evaluate -- treated as
    not covered, never as covered by default)."""
    if not sentence_toks:
        return 0.0
    total = sum(idf.get(w, 1.0) for w in sentence_toks)
    if total <= 0:
        return 0.0
    covered = sum(idf.get(w, 1.0) for w in sentence_toks if w in idf)
    return covered / total


def _source_idf(sources: List[str]) -> Dict[str, float]:
    """idf(w) = ln(1 + N / df(w)), N = number of source documents, df(w) = how many of them
    contain w -- the exact formula attribution_check.py's own _idf already uses for excerpt
    relevance scoring, applied here to whole source documents instead of passage windows."""
    doc_toks = [_toks(s) for s in sources if s and s.strip()]
    n = max(1, len(doc_toks))
    df: Dict[str, int] = {}
    for toks in doc_toks:
        for w in toks:
            df[w] = df.get(w, 0) + 1
    return {w: math.log(1 + n / c) for w, c in df.items()}


def provenance_spans(
    final_text: str,
    sources: List[str],
    *,
    threshold: float = SOURCED_COVERAGE_THRESHOLD,
) -> List[Span]:
    """Classifies every sentence of `final_text` into exactly three tiers, by formula, no LLM
    call:

      QUOTED    -- Q(S) = 1 (see _is_verbatim_match): the sentence is a near-verbatim match of
                   something in `sources`.
      SOURCED   -- Q(S) = 0 and C(S) >= threshold (see _coverage_score): the sentence is the
                   model's own wording, but its distinctive vocabulary substantially overlaps
                   the source corpus's vocabulary.
      UNSOURCED -- Q(S) = 0 and C(S) < threshold: the sentence shares little or no distinctive
                   vocabulary with any available source.

    `sources` is plain text: whatever was actually available when this turn was generated
    (retrieved case law/legislation snippets for legal personas, exhibit texts, or the client's
    own prior messages for intake) -- built by the caller, since what counts as "a source" is
    different per persona. An empty `sources` list means every sentence falls to UNSOURCED
    (C(S) is 0 against an empty corpus by construction), which is correct: nothing was available
    to draw on.

    Every sentence gets a tier; none are ever omitted -- an earlier version of this module
    omitted an "unaddressed" verdict, which produced large uncolored stretches of real answers
    indistinguishable from "checked and clean" (found live, during testing). There is no such
    gap here: the formula always returns a definite classification.

    Deterministic and free: no network call, no judge, so no "fails open on exception" story is
    needed for the classification step itself -- the only failure mode is a bug, not a flaky
    dependency. Still wrapped by the caller in the same try/except discipline as every other
    confidence.py function, as defence in depth."""
    spans: List[Span] = []
    if not final_text:
        return spans
    sentences = _candidate_sentences(final_text)
    if not sentences:
        return spans
    sources = [_clean_source(s) for s in (sources or [])]
    idf = _source_idf(sources)
    norm_sources = [_normalize(s) for s in sources]
    for sentence in sentences:
        if _is_verbatim_match(sentence, norm_sources, normalized=True):
            spans.append(Span(quote=sentence, tier="QUOTED", source="provenance",
                               why="near-verbatim match of an available source"))
            continue
        score = _coverage_score(_toks(sentence), idf)
        if score >= threshold:
            spans.append(Span(quote=sentence, tier="SOURCED", source="provenance",
                               why=f"{score:.0%} of its distinctive terms appear in the available sources"))
        else:
            spans.append(Span(quote=sentence, tier="UNSOURCED", source="provenance",
                               why="no available source shares this sentence's distinctive vocabulary"))
    return spans


# ---------------------------------------------------------------------------
# Citation links -- QUOTED when a link resolves, zero computation (exact, not scored)
# ---------------------------------------------------------------------------

_LEGAL_REF_ANCHOR_RE = re.compile(
    r'<a href="([^"]+)" class="legal-ref (law|jurisprudence)"[^>]*>(.*?)</a>', re.S
)


def quoted_citation_spans(text: str, search_results) -> List[Span]:
    """Citation links already embedded as `<a class="legal-ref ...">` HTML by
    legal_citations.format_legal_citation_links (which runs INSIDE _finalize_legal_response,
    before this function or anything else ever sees final_text) get QUOTED when their href
    resolves to one of the turn's actually-retrieved sources -- an exact, zero-cost check
    (classify_href), not a formula needing a threshold. A link that does NOT resolve gets no
    span at all: this module now has exactly three tiers with no separate "flagged as wrong"
    case, so an unresolved link's surrounding sentence is classified like any other text by
    provenance_spans instead.

    Fails open: any exception is logged and simply omits this part of the signal."""
    spans: List[Span] = []
    if not text:
        return spans
    try:
        from legal_citations import allowed_url_set, classify_href
        allowed = allowed_url_set(search_results)
        links = [(href, re.sub(r"<[^>]+>", "", raw_label).strip())
                 for href, _kind, raw_label in _LEGAL_REF_ANCHOR_RE.findall(text)]
        # Markdown-form citations too: UK answers can carry "[label Jurisprudence](url)" instead
        # of legal-ref <a> HTML (found live; the UK regression suite accepts both forms).
        links += [(href, _plain_for_match(raw_label).strip("*_ ")) for raw_label, href in _MD_HTTP_LINK_RE.findall(text)]
        seen = set()
        for href, label in links:
            if not label or (href, label) in seen:
                continue
            seen.add((href, label))
            if classify_href(href, allowed) == "ok":
                spans.append(Span(quote=label, tier="QUOTED", source="citation",
                                   why="citation link resolves to a retrieved source", ref=href))
    except Exception as e:
        logging.warning("confidence.quoted_citation_spans failed (no spans produced): %s", e)
    return spans


# ---------------------------------------------------------------------------
# Payload assembly
# ---------------------------------------------------------------------------

def build_payload(
    persona: str,
    model: Optional[str],
    spans: List[Span],
    *,
    tokens: Optional[dict] = None,
    checked: bool = False,
    legend: str = "provenance",
) -> dict:
    """`coverage` is the honesty field: "none" means no check ran at all (never shown as clean
    unmarked text implying classification happened), "partial" means one of (provenance
    classification, entropy telemetry) ran, "full" means both. `checked` is passed by the caller
    to distinguish "ran and found nothing to flag" (spans == [] but checked == True) from
    "never ran" (checked == False, "none") -- an empty span list alone cannot tell those apart."""
    spans = spans or []
    if checked and tokens:
        coverage = "full"
    elif checked or tokens:
        coverage = "partial"
    else:
        coverage = "none"
    return {
        "type": "confidence",
        "v": SCHEMA_VERSION,
        "persona": persona,
        "model": model,
        "coverage": coverage,
        "legend": legend,
        "spans": [s.to_dict() for s in spans],
        "tokens": tokens,
    }


def to_frame(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
