# -*- coding: utf-8 -*-
"""Quotation checks against the case's own exhibits.

The blockquote gate in legal_citations only trusts retrieved legal text, so an answer that quotes
a witness statement or email either lost the quote or was never checked. And a phrase put in
quote marks inline ("still being "juggled"" for an email that says "juggling the mezzanine
rota") was never checked at all. These helpers treat the exhibits as a source:

  * exhibit_texts(state)            the full text of each exhibit held for this turn;
  * find_exhibit_misquotes(...)     inline quoted phrases that nearly match exhibit wording but are
                                    not verbatim (a changed word form, a dropped or added word);
  * unquote_exhibit_misquotes(...)  last-resort fix: remove only the quote marks around those, so
                                    the answer never presents altered wording as the document's own.

A phrase with no near match in any exhibit (a column heading, a filename, a drafted clause) is not
a claim about an exhibit and is left alone. No server imports: pure helpers.
"""

from __future__ import annotations

import re
from typing import List, Tuple

_NORMALIZE = str.maketrans(
    {"‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", " ": " "}
)
_TOKEN = re.compile(r"[a-z0-9]+")
_QUOTED = re.compile(r"[“\"]([^“”\"\n]{2,200})[”\"]")
_TAG = re.compile(r"(<[^>]+>)")
_SUFFIXES = ("ments", "ment", "ings", "ing", "ions", "ion", "ers", "er", "ed", "es", "s", "ly")
_STOP = frozenset("a an and as at be by for in is it of on or the to was were with".split())
_NEAR_MISS_SCORE = 0.75


def exhibit_texts(state) -> List[str]:
    """Full text of each case document held on the session this turn. Never raises."""
    out: List[str] = []
    for entry in (getattr(state, "case_document_cache", None) or {}).values():
        if isinstance(entry, dict):
            text = entry.get("text")
            if isinstance(text, str) and text.strip():
                out.append(text)
    return out


def _stem(word: str) -> str:
    for suffix in _SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 4:
            word = word[: -len(suffix)]
            break
    return word[:-1] if len(word) > 4 and word.endswith("e") else word


class _Index:
    """Exhibit tokens with their positions, so a matching stretch can be quoted back."""

    def __init__(self, texts: List[str]):
        self.norm = "\n\n".join(texts).translate(_NORMALIZE)  # same length as the source text
        self.tokens = [(m.group(), m.start(), m.end()) for m in _TOKEN.finditer(self.norm.lower())]
        self.words = [t[0] for t in self.tokens]
        self.stems = [_stem(w) for w in self.words]

    def best_window(self, quoted_tokens: List[str]) -> Tuple[float, bool, int]:
        """(best score, whether some window matches exactly, start index of the best window)."""
        n = len(quoted_tokens)
        q_stems = [_stem(t) for t in quoted_tokens]
        best, best_at, exact = 0.0, 0, False
        for i in range(0, len(self.words) - n + 1):
            hits = sum(1 for k in range(n) if self.stems[i + k] == q_stems[k])
            score = hits / n
            if score > best:
                best, best_at = score, i
            if score == 1.0 and self.words[i : i + n] == quoted_tokens:
                exact = True
                break
        return best, exact, best_at

    def dropped_word_window(self, quoted_tokens: List[str]) -> int:
        """Start index of an exhibit stretch that is the quote plus ONE extra interior word
        (the quote dropped it: "adjustments fully implemented" for "adjustments were fully
        implemented"), or -1. Equal-length matches are handled by best_window."""
        n = len(quoted_tokens)
        q_stems = [_stem(t) for t in quoted_tokens]
        for i in range(0, len(self.stems) - n):
            window = self.stems[i : i + n + 1]
            if any(window[:k] + window[k + 1 :] == q_stems for k in range(1, n)):
                return i
        return -1

    def snippet(self, start: int, n: int) -> str:
        return self.norm[self.tokens[start][1] : self.tokens[start + n - 1][2]]


def _near_miss(quoted: str, index: _Index) -> Tuple[bool, str]:
    """(True, the exhibit's wording) when `quoted` nearly matches an exhibit but is not verbatim."""
    tokens = _TOKEN.findall(quoted.translate(_NORMALIZE).lower())
    if not tokens or len(tokens) > 30 or len(index.words) < len(tokens):
        return False, ""
    if all(t in _STOP or len(t) < 4 for t in tokens):
        return False, ""
    score, exact, at = index.best_window(tokens)
    if exact:
        return False, ""
    if score >= _NEAR_MISS_SCORE:
        return True, index.snippet(at, len(tokens))
    if len(tokens) >= 3:
        dropped = index.dropped_word_window(tokens)
        if dropped >= 0:
            return True, index.snippet(dropped, len(tokens) + 1)
    return False, ""


def _prose_segments(text: str):
    """Yield (is_prose, piece): HTML tags and blockquote lines are not inline prose — the
    blockquote gate owns those, and quote characters inside a tag are attributes."""
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith(">"):
            yield False, line
            continue
        for piece in _TAG.split(line):
            if piece:
                yield (not piece.startswith("<")), piece


def find_exhibit_misquotes(text: str, texts: List[str]) -> List[Tuple[str, str]]:
    """[(quoted phrase, what the exhibit actually says)] for each near-miss, first seen order."""
    if not text or not texts:
        return []
    index = _Index(texts)
    found: List[Tuple[str, str]] = []
    seen = set()
    for is_prose, piece in _prose_segments(text):
        if not is_prose:
            continue
        for m in _QUOTED.finditer(piece):
            quoted = m.group(1)
            if quoted in seen:
                continue
            seen.add(quoted)
            miss, source = _near_miss(quoted, index)
            if miss:
                found.append((quoted, source))
    return found


def unquote_exhibit_misquotes(text: str, texts: List[str]) -> str:
    """Remove the quote marks (only) around inline phrases that nearly match an exhibit but are
    not verbatim. The wording stays; it just stops claiming to be the document's own words."""
    if not text or not texts:
        return text
    index = _Index(texts)
    cache = {}

    def repl(m: "re.Match") -> str:
        quoted = m.group(1)
        if quoted not in cache:
            cache[quoted] = _near_miss(quoted, index)[0]
        return quoted if cache[quoted] else m.group(0)

    return "".join(
        _QUOTED.sub(repl, piece) if is_prose else piece for is_prose, piece in _prose_segments(text)
    )
