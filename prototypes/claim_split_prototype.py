# -*- coding: utf-8 -*-
"""PROTOTYPE -- throwaway, not wired into the real pipeline.

Question this answers: if a sentence is split into atomic claim units at connectives
("but", "however", "therefore", ...) BEFORE provenance scoring, does per-unit
classification actually diverge from whole-sentence classification in cases that matter --
i.e. does a compound sentence hide a QUOTED half behind an UNSOURCED half (or vice versa)
when scored as one unit? If splitting never changes the verdict on realistic sentences,
it isn't worth the complexity it would add to confidence.py's _candidate_sentences.

This module is pure (no I/O, no terminal code) so the split function can be lifted
directly into confidence.py if the answer is "yes, it matters" -- see NOTES.md for the
verdict once filled in. It imports confidence.py's real scoring primitives rather than
reimplementing them, so what's being tested is the SPLIT, not a second copy of the formula.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from confidence import (
    SOURCED_COVERAGE_THRESHOLD,
    _coverage_score,
    _is_verbatim_match,
    _normalize,
    _source_idf,
    _toks,
)

# English connectives plausible in UK/legal prose. Dr. Kim's VeriLENS advice used Korean
# connectives (다만/그러나/따라서 -- "however/but/therefore"); this is the English equivalent
# set, not a translation of a fixed list -- chosen for what actually shows up in legal
# reasoning (concession, contrast, consequence), same semantic categories Dr. Kim named.
_CONNECTIVES = [
    "provided that", "except that", "on the other hand", "notwithstanding",
    "subject to", "save that", "nevertheless", "nonetheless", "accordingly",
    "however", "although", "therefore", "whereas", "meanwhile", "though",
    "unless", "hence", "thus", "while", "but", "yet",
]

# A connective only marks a genuine clause boundary when it follows a comma/semicolon --
# "but" mid-noun-phrase ("nothing but the truth") never does. Sorted longest-first so
# "provided that" matches before a bare "that" could ever be considered.
_CONN_RE = re.compile(
    r"[,;]\s+(" + "|".join(re.escape(c) for c in sorted(_CONNECTIVES, key=len, reverse=True)) + r")\b,?\s*",
    re.IGNORECASE,
)


def split_into_claims(sentence: str) -> List[str]:
    """Breaks one (already-cleaned) sentence into atomic claim units at connectives. The
    connective word is kept at the START of the unit it introduces -- that's the clause it
    semantically belongs to ("X, but Y" -> ["X", "but Y"], not ["X", "Y"], so a reader can
    still see why the second unit exists). Returns [sentence] unchanged when no connective
    boundary is found, which is the common case -- most sentences aren't compound.

    UNGUARDED: see the adversarial batch findings in NOTES.md. This version splits on
    EVERY comma-preceded connective, including a parenthetical aside ("X, although Y, Z")
    that is not a real clause boundary -- confirmed to produce confidently wrong tiers on
    real examples. Kept here for comparison; split_into_claims_guarded is the fix."""
    parts: List[str] = []
    last = 0
    for m in _CONN_RE.finditer(sentence):
        head = sentence[last:m.start()].strip(" ,;")
        if head:
            parts.append(head)
        last = m.start(1)  # next unit starts AT the connective word, not after it
    tail = sentence[last:].strip(" ,;")
    if tail:
        parts.append(tail)
    return parts if len(parts) > 1 else [sentence]


# These five subordinators are the ones the adversarial batch caught opening a reduced,
# verbless aside ("although serious,", "while sympathetic...,") rather than a genuine
# subordinate clause -- confirmed on "although"/"while"/"yet" in NOTES.md. The other
# connectives (but, however, therefore, ...) were never observed misfiring this way, even
# when an unrelated later comma appears (e.g. a nested aside inside the clause they open),
# so they are left unguarded rather than adding a check with no observed failure to fix.
_RISKY_SUBORDINATORS = frozenset({"although", "though", "while", "whereas", "yet"})

# SUPERSEDED -- kept only as a comment for the record (see NOTES.md "round 2"): a
# verb-presence check ("does the window contain a verb-like word?") was tried first and
# broken by reduced PAST-PARTICIPLE asides -- "although delayed," / "while advised
# against it," -- where the "-ed" ending looks exactly like a real verb despite there
# being no subject at all. "although delayed" and "although the claimant appealed" both
# contain what looks like a verb; only one of them is a genuine clause.
#
# The actual discriminator is SUBJECT presence, not verb presence: a reduced aside is
# reduced precisely because it omits its own subject ("[being] delayed", "[though it was]
# sympathetic", "[yet] again") -- it launches straight into an adjective, adverb, or bare
# participle. A genuine subordinate clause always opens with its own subject first (a
# determiner, pronoun, quantifier, or proper noun) before anything else.
_SUBJECT_STARTERS = frozenset(
    "the a an this these that those his her their its our your my many some most each "
    "every any no such he she it they we you i one".split()
)
_WORD_FINDER = re.compile(r"[a-zA-Z']+")


def _opens_subjectless_aside(sentence: str, after: int) -> bool:
    """True when the text from `after` (just past a matched connective) to the next comma
    does NOT open with its own subject -- the structural signature of a reduced
    parenthetical aside ("although delayed,", "while sympathetic...,") rather than a
    genuine subordinate clause ("although the claimant appealed,"). A connective that
    runs all the way to the sentence end with no further comma is never treated as an
    aside (nothing closes it, and a real clause is free to contain its own list commas
    before the sentence ends -- see the 'genuine clause, internal list comma' case)."""
    next_comma = sentence.find(",", after)
    if next_comma == -1:
        return False
    window_words = _WORD_FINDER.findall(sentence[after:next_comma])
    if not window_words:
        return True
    first = window_words[0].lower()
    if first in _SUBJECT_STARTERS:
        return False
    # A capitalised first word this far into the sentence is almost certainly a proper
    # noun acting as a subject ("while Parliament intended..."), not an adjective.
    return not window_words[0][0].isupper()


def split_into_claims_guarded(sentence: str) -> List[str]:
    """split_into_claims plus one guard: a _RISKY_SUBORDINATOR is skipped as a split point
    when _opens_subjectless_aside says it's opening a reduced parenthetical ("X, although
    Y, Z") rather than its own independent/subordinate clause. Everything else about the
    splitting logic -- which connectives count, where the boundary falls, the single-unit
    fallback -- is identical to split_into_claims."""
    parts: List[str] = []
    last = 0
    for m in _CONN_RE.finditer(sentence):
        word = m.group(1).lower()
        if word in _RISKY_SUBORDINATORS and _opens_subjectless_aside(sentence, m.end()):
            continue  # parenthetical aside -- not a genuine clause boundary, skip it
        head = sentence[last:m.start()].strip(" ,;")
        if head:
            parts.append(head)
        last = m.start(1)
    tail = sentence[last:].strip(" ,;")
    if tail:
        parts.append(tail)
    return parts if len(parts) > 1 else [sentence]


def classify(
    text: str,
    norm_sources: List[str],
    idf: Dict[str, float],
    threshold: float = SOURCED_COVERAGE_THRESHOLD,
) -> Tuple[str, Optional[float]]:
    """The exact three-tier rule provenance_spans uses in confidence.py, exposed standalone
    so the TUI can call it on a whole sentence AND on each of its claim units. Returns
    (tier, coverage_score_or_None) -- score is None for QUOTED since C(S) was never computed."""
    if _is_verbatim_match(text, norm_sources, normalized=True):
        return "QUOTED", None
    score = _coverage_score(_toks(text), idf)
    tier = "SOURCED" if score >= threshold else "UNSOURCED"
    return tier, score


def build_idf(sources: List[str]) -> Dict[str, float]:
    return _source_idf(sources)


def normalize_sources(sources: List[str]) -> List[str]:
    return [_normalize(s) for s in sources]
