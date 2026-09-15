# -*- coding: utf-8 -*-
"""Okapi BM25 — term-frequency/inverse-document-frequency ranking with document-length
normalization (Robertson & Walker). A standalone, from-scratch implementation of the published
algorithm, written after finding that the "SCL Core" tutorial build (a separate, unshipped
download, not part of this repo) has its own BM25 class compiled into `the_sandbox.pyc` — no
source available, and not something this repo imports from or depends on. Its usage (peak-scaled
normalized scores via `.scores_normalised()`) is the reference this module's interface follows;
no code from that build is reused here.

Not wired into any retrieval path yet. `uk_legal_mcp/scoring.py`'s `fill_missing_scores` currently
backfills relevance with a cruder query-keyword-hit-density heuristic
(`matches / len(query_terms)`, ignoring term rarity and document length) — a natural place this
could plug in later, but that integration is a separate decision, not made here.

`rrf_rank_scores` below is a faithful port of the SCL Core reference's `_rrf_rank_scores` /
`_positive_rank_map` (the_server.py:3170-3198) — Reciprocal Rank Fusion, combining BM25 scores
with a second, index-aligned score list (e.g. embedding/cosine similarity) into one fused
ranking. Also unwired: what BM25 actually gets fused *with* in this repo is the same open
decision as where BM25 itself plugs in.

Pure module: stdlib only (re, math, collections), no server/session imports, so it can be tested
and reasoned about in isolation.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

_DEFAULT_K1 = 1.5
_DEFAULT_B = 0.75

# Unicode letters/numbers as one run. Matches the tokenization style already used elsewhere in
# this repo for lexical text (see the_server.py's _lexical_words), minus the CJK ranges — none
# of this repo's current corpora (PH/UK case law, case-document chunks) need them, and adding
# ranges no caller exercises just widens the surface for false-positive matches.
_TOKEN_PATTERN = re.compile(r"[^\W\d_]+|\d+", re.UNICODE)


def _tokenize(text: str) -> List[str]:
    return _TOKEN_PATTERN.findall((text or "").lower())


class BM25:
    """Scores a fixed corpus of texts against arbitrary queries.

    Usage mirrors the SCL Core reference: construct once per corpus, call `.scores(query)` (or
    `.scores_normalised(query)`) once per query — construction is the expensive step (tokenizes
    every document and builds the IDF table), scoring is cheap.
    """

    def __init__(self, corpus: Sequence[str], *, k1: float = _DEFAULT_K1, b: float = _DEFAULT_B) -> None:
        self.k1 = k1
        self.b = b
        self._doc_term_freq: List[Counter] = [Counter(_tokenize(text)) for text in corpus]
        self._doc_lengths: List[int] = [sum(freq.values()) for freq in self._doc_term_freq]
        self.size = len(corpus)
        self._avg_doc_length = (sum(self._doc_lengths) / self.size) if self.size else 0.0

        doc_freq: Counter = Counter()
        for freq in self._doc_term_freq:
            doc_freq.update(freq.keys())

        # Robertson-Sparck-Jones IDF with a +1 inside the log so a term present in every
        # document floors at 0 rather than going negative (the standard BM25+ fix — matters
        # most on a small or near-duplicate-heavy corpus, e.g. a single case-document bundle).
        self._idf: Dict[str, float] = {
            term: math.log((self.size - df + 0.5) / (df + 0.5) + 1) for term, df in doc_freq.items()
        }

    def scores(self, query: str) -> List[float]:
        """Raw BM25 score for `query` against every corpus document, in corpus order."""
        query_terms = _tokenize(query)
        return [self._score_doc(query_terms, i) for i in range(self.size)]

    def scores_normalised(self, query: str) -> List[float]:
        """`scores()` divided by its own maximum, so the top hit is 1.0 — peak-scaled rather
        than min-max, so one dominant hit doesn't compress every other score toward zero."""
        raw = self.scores(query)
        peak = max(raw, default=0.0)
        if peak <= 0.0:
            return [0.0] * len(raw)
        return [max(0.0, score / peak) for score in raw]

    def _score_doc(self, query_terms: List[str], doc_index: int) -> float:
        doc_length = self._doc_lengths[doc_index]
        if doc_length == 0:
            return 0.0
        term_freq = self._doc_term_freq[doc_index]

        score = 0.0
        for term in query_terms:
            idf = self._idf.get(term)
            tf = term_freq.get(term) if idf else None
            if not idf or not tf:
                continue  # term absent from the corpus, or from this document
            denom = tf + self.k1 * (1 - self.b + (self.b * doc_length) / (self._avg_doc_length or 1))
            score += idf * ((tf * (self.k1 + 1)) / denom)
        return score


def rank(corpus: Sequence[str], query: str, top_n: Optional[int] = None) -> List[int]:
    """Convenience one-shot: corpus indices sorted by BM25 score, highest first. Builds a fresh
    BM25 index each call — fine for a one-off ranking, wasteful if scoring many queries against
    the same corpus (construct `BM25(corpus)` once and reuse it for that case instead)."""
    scores = BM25(corpus).scores(query)
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[:top_n] if top_n is not None else order


def _positive_rank_map(scores: Sequence[float]) -> Dict[int, int]:
    """`scores` indices ranked 1..N by descending value, *excluding* any index whose score is
    zero, negative, or non-finite entirely — such an index gets no rank at all rather than being
    pushed to the bottom, so it contributes nothing when `rrf_rank_scores` looks it up. Ties
    break by original index (lower index wins), matching the reference exactly."""
    ranked: List[Tuple[int, float]] = []
    for idx, score in enumerate(scores):
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(value) or value <= 0.0:
            continue
        ranked.append((idx, value))
    ranked.sort(key=lambda item: (-item[1], item[0]))
    return {idx: rank for rank, (idx, _score) in enumerate(ranked, 1)}


def rrf_rank_scores(embedding_scores: Sequence[float], bm25_scores: Sequence[float], k: int = 60) -> List[float]:
    """Reciprocal Rank Fusion: combines two index-aligned score lists (e.g. embedding/cosine
    similarity and `BM25.scores()`) into one fused ranking, each item scored
    `1/(k + rank_in_list)` summed across whichever list(s) it has a positive rank in, then
    peak-normalized. `k=60` is RRF's standard damping constant — large enough that rank 1 vs
    rank 2 in one list can't swamp the other list's contribution. An item ranked well in *both*
    lists outranks one ranked well in only one; an item absent (or non-positive) in a list
    simply contributes 0 for that list, not a penalty.

    The two lists need not be the same length — the result is sized to the longer one, per the
    reference's `size = max(len(embedding_scores), len(bm25_scores))`.
    """
    size = max(len(embedding_scores), len(bm25_scores))
    if size <= 0:
        return []
    embedding_ranks = _positive_rank_map(embedding_scores)
    bm25_ranks = _positive_rank_map(bm25_scores)

    raw: List[float] = []
    for idx in range(size):
        score = 0.0
        if idx in embedding_ranks:
            score += 1.0 / (k + embedding_ranks[idx])
        if idx in bm25_ranks:
            score += 1.0 / (k + bm25_ranks[idx])
        raw.append(score)

    peak = max(raw) if raw else 0.0
    if peak <= 0.0:
        return [0.0] * size
    return [round(score / peak, 6) for score in raw]
