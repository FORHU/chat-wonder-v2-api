# -*- coding: utf-8 -*-
"""Attributed-claim verification: does any case document record what the answer says a party said?

ILL-ARB C1 (critical): an answer weighed a witness statement against "Solway's suggestion that
Cairngorm had no FAT evidence at all", a position no document records. The words of such a sentence
all appear somewhere in the documents, so a word-overlap check cannot catch it; the question is one
of meaning (did THEY take that position?).

Sentences that attribute a statement, position or admission to a named party, witness or document are
picked out deterministically. For each, the most relevant excerpts of the exhibits are retrieved
(headed by the document's own author/type lines) and a judge reads the claim against ITS excerpts and
must quote the words that support it. A first, whole-bundle version caught every fabrication but also
flagged ~60% of genuine attributions (5 of 8, identical on every run): a small model cannot find the
needle in a whole case file. Judging each claim against its own excerpts, with code-side overrides
below, is the fix; a probe with known-false attributions confirmed its precision (see
attribution_check_enabled).

Safe by construction: it fails open (any error, timeout or unparseable reply means "nothing
unsupported"); a flag is dropped when the claim is near-verbatim in an exhibit or the judge quotes
support while calling it unsupported; it is skipped without exhibits. No server imports: the judge
is injected.
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

_MAX_CLAIMS = 40
_EXCERPT_DOCS = 5  # documents whose best passage is shown for each claim
_WINDOW_SENTENCES = 3
_EXCERPT_CHARS = 900
_HEADER_CHARS = 400
_VERBATIM_COVERAGE = 0.95  # a claim this fully contained in one stretch of an exhibit is never flagged

# An attribution is a NAMED subject directly followed by an attribution verb (then "that" or a
# determiner) or noun ("Solway's suggestion that ..."). A loose verb match is mostly noise in legal
# prose: "Equality Act claims", "state expressly which sums", "claimed under the Act".
_SUBJECT = (
    r"(?:(?:Mr|Ms|Mrs|Miss|Dr)\.?\s+[A-Z][a-z]+(?:['â€™]s)?|[A-Z][a-z]{2,}(?:['â€™]s)?"
    r"|[Tt]he (?:Claimant|Respondent|claimant|respondent|employer|employee|witness)(?:['â€™]s)?)"
)
_ADVERB = r"(?:\s+(?:also|further|expressly|now|still|consistently|repeatedly))?"
_ATTRIB = re.compile(
    rf"\b(?P<subject>{_SUBJECT}){_ADVERB}\s+"
    r"(?:(?:says?|said|argues?|argued|contends?|contended|asserts?|asserted|suggests?|suggested|alleges?|"
    r"alleged|maintains?|maintained|submits?|submitted|claims?|claimed|accepts?|accepted|admits?|admitted|"
    r"denies|denied|concedes?|conceded|states?|stated|insists?|insisted)\s+(?:that|the|a|an|it|he|she|there|no|"
    r"his|her|their|they)\b"
    r"|(?:suggestion|contention|assertion|allegation|argument|position|case|account|evidence)\s+(?:is\s+)?that\b)"
)
# Subjects that are law, not a party, witness or document of the case.
_NOT_A_PARTY = re.compile(r"\b(?:Act|Acts|Regulations|Rules|Order|Code|Convention|Section|Article|Tribunal|Court|Parliament)$")
# "may argue", "could say", "should state": the answer's own hypothesis or drafting, not an attribution.
_MODAL_BEFORE = re.compile(r"\b(?:may|might|could|can|should|would|will|must|shall|to)\s+(?:\w+\s+)?$", re.I)
# A conditional or hypothetical ("If the Claimant alleges...", "where the Claimant says...") describes what
# WOULD follow, not what someone has said. Live false positives: advice sentences and schedule line items.
_CONDITIONAL_LEAD = re.compile(r"^\W*(?:if|where|whether|when|unless|should|so far as|to the extent)\b", re.I)
_CONDITIONAL_BEFORE = re.compile(r"\b(?:if|where|whether|when|unless|as|so far as|to the extent that)\s+(?:\w+\s+){0,3}$", re.I)
_HYPOTHETICAL_AFTER = re.compile(r"\b(?:would|could|might|may|should)\b", re.I)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"â€œ(])")
_PASSAGE_SPLIT = re.compile(r"(?<=[.?!])\s+|\n{2,}")

_STOP = frozenset(
    "a an and are as at be by for from has have in is it of on or that the their this to was were which with "
    "should would could may can not no must also any each both same into than then there these those its his her "
    "him she he they them said says say".split()
)
_SYSTEM = "You verify attributions in a legal answer against excerpts of the case documents. Return ONLY valid JSON."
_INSTRUCTIONS = (
    "Each numbered CLAIM is a sentence from the answer, followed by the EXCERPTS of the case documents most "
    "relevant to it; each excerpt is headed by its document's own title and author lines. For each claim that "
    "attributes a statement, position, argument, admission or denial to a party, witness or document of THIS "
    "case (not to a statute or a reported case), decide from its excerpts whether a document records that "
    "person, party or document saying, arguing, suggesting, admitting or denying that, on a fair reading: a "
    "witness statement is the voice of its author, and a paraphrase counts. A true underlying fact is not "
    "enough, and neither are the same words used for something else. If the excerpts do not show them taking "
    "that position, the claim is unsupported, even if the idea is plausible.\n\n"
    "Verdict 'not_an_attribution' for statements that are the answer's own argument, advice or hypothesis "
    "(for example 'may argue', 'could say', 'if the Claimant alleges...', 'where the Respondent says...'), "
    "descriptions of what a schedule or document should contain, and statements about a statute or reported "
    "case. Only judge a sentence that reports what someone HAS said, argued, admitted or denied. If you are "
    "unsure, use 'supported'.\n\n"
    'Return ONLY: {"results": [{"n": <claim number>, "verdict": "supported" | "unsupported" | '
    '"not_an_attribution", "quote": "<the exact words copied from an excerpt that show it was said; empty if '
    'unsupported>", "why": "<=20 words"}]} with one entry per claim.'
)


def attribution_check_enabled() -> bool:
    """ON by default; LEGAL_ATTRIBUTION_CHECK=false switches it off.

    It was held off until a probe (known-false attributions mixed among genuine ones, real judge, real
    exhibits) showed acceptable precision. History of that probe: the whole-bundle judge flagged ~60% of
    genuine attributions (5 of 8); per-claim excerpts on the stronger model cut that to 1 of 8, which
    turned out to be a retrieval miss (the Claimant's own statement was never shown); showing the best
    passage of each relevant document fixed it. Final probe: 3 repeats x 2 passes, 5/5 fabrications
    caught every time, 0 genuine sentences flagged. The sample is small (5 fabrications, 8 genuine
    sentences, one case), so the logged verdicts of real runs are the ongoing check."""
    return os.getenv("LEGAL_ATTRIBUTION_CHECK", "true").strip().lower() not in ("0", "false", "no", "off")


def _plain(line: str) -> str:
    line = re.sub(r"<[^>]+>", "", line)
    line = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", line)
    return re.sub(r"[*_`]+", "", line).strip()


def candidate_attributions(text: str, limit: int = _MAX_CLAIMS) -> List[str]:
    """Sentences that attribute something to a named party, witness or document, in order."""
    return [sentence for sentence, _claim, _subject in _candidates(text, limit)]


def _candidates(text: str, limit: int = _MAX_CLAIMS) -> List[Tuple[str, str, str]]:
    """[(sentence, claim, subject)]. The claim is the attribution itself, cut from the subject onward:
    in "A is more probative than Brightwell's suggestion that X" or "the minutes record Y, while the
    Claimant says Z" the judge is asked about "Brightwell's suggestion that X" / "the Claimant says Z",
    not about the answer's own weighing around it (which a judge reads as the answer's own argument)."""
    out: List[Tuple[str, str, str]] = []
    for line in (text or "").splitlines():
        stripped = line.lstrip()
        if not stripped or stripped[0] in ">|#":
            continue  # quotations are checked elsewhere; tables and headings are not claims
        for sentence in _SENTENCE_SPLIT.split(_plain(line)):
            sentence = sentence.strip()
            if not 40 <= len(sentence) <= 420 or _CONDITIONAL_LEAD.match(sentence):
                continue
            for m in _ATTRIB.finditer(sentence):
                if (
                    not _NOT_A_PARTY.search(m.group("subject"))
                    and not _MODAL_BEFORE.search(sentence[max(0, m.start() - 25): m.start()])
                    and not _CONDITIONAL_BEFORE.search(sentence[max(0, m.start() - 40): m.start()])
                    and not _HYPOTHETICAL_AFTER.search(sentence[m.end(): m.end() + 60])
                ):
                    out.append((sentence, sentence[m.start():].strip(), m.group("subject")))
                    if len(out) >= limit:
                        return out
                    break
    return out


# --- retrieval: the excerpts a claim is judged against -------------------------------------------

def _stem(w: str) -> str:
    for suf in ("ments", "ment", "ings", "ing", "ions", "ion", "ers", "er", "ed", "es", "s", "ly"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            w = w[: -len(suf)]
            break
    return w[:-1] if len(w) > 4 and w.endswith("e") else w


def _toks(s: str) -> List[str]:
    return [_stem(w) for w in re.findall(r"[a-z0-9]+", s.lower()) if w not in _STOP and len(w) > 2]


@dataclass
class _Passage:
    doc: int
    header: str
    text: str
    toks: Set[str]


def _passages(exhibits: List[str]) -> List[_Passage]:
    out: List[_Passage] = []
    for i, doc in enumerate(exhibits, 1):
        header = re.sub(r"\s+", " ", doc[:_HEADER_CHARS]).strip()
        sents = [s.strip() for s in _PASSAGE_SPLIT.split(doc) if s and s.strip()]
        for k in range(len(sents)):
            window = " ".join(sents[k: k + _WINDOW_SENTENCES])[:_EXCERPT_CHARS]
            toks = set(_toks(window))
            if toks:
                out.append(_Passage(i, header, window, toks))
    return out


def _idf(passages: List[_Passage]) -> Dict[str, float]:
    import math

    df: Dict[str, int] = {}
    for p in passages:
        for t in p.toks:
            df[t] = df.get(t, 0) + 1
    n = max(1, len(passages))
    return {t: math.log(1 + n / c) for t, c in df.items()}


def _clause(sentence: str) -> str:
    m = re.search(
        r"\b(?:says?|said|argues?|contends?|asserts?|suggests?|alleges?|maintains?|submits?|claims?|accepts?|"
        r"admits?|denies|concedes?|states?|insists?|suggestion|contention|assertion|allegation|argument|"
        r"position|case|account|evidence)\b\s*(?:(?:is\s+)?that)?",
        sentence,
        re.I,
    )
    return sentence[m.end():] if m else sentence


def _coverage(clause_toks: Set[str], window_toks: Set[str]) -> float:
    return len(clause_toks & window_toks) / len(clause_toks) if clause_toks else 0.0


def _excerpts(sentence: str, subject: str, passages: List[_Passage], idf: Dict[str, float]) -> Tuple[List[_Passage], float]:
    """(the best few passages for this claim, the best single-passage coverage of its clause)."""
    clause = set(_toks(_clause(sentence)))
    subj = set(_toks(subject))
    scored: List[Tuple[float, _Passage]] = []
    best_cov = 0.0
    for p in passages:
        score = sum(idf.get(t, 0.0) for t in clause & p.toks)
        score += 0.5 * sum(idf.get(t, 0.0) for t in subj & (p.toks | set(_toks(p.header))))
        best_cov = max(best_cov, _coverage(clause, p.toks))
        if score > 0:
            scored.append((score, p))
    # The best passage of each of the most relevant DOCUMENTS, not the top windows overall: overlapping
    # windows from one paragraph used to fill every slot, so the Claimant's own statement (rank 16 by
    # score, rank 1 within its own document) was never shown and a genuine claim was judged unsupported.
    best_per_doc: Dict[int, Tuple[float, _Passage]] = {}
    for score, p in scored:
        if p.doc not in best_per_doc or score > best_per_doc[p.doc][0]:
            best_per_doc[p.doc] = (score, p)
    ranked = sorted(best_per_doc.values(), key=lambda sp: -sp[0])
    return [p for _score, p in ranked[:_EXCERPT_DOCS]], best_cov


def _parse(raw: str, n_claims: int) -> List[Tuple[int, str, str, str]]:
    """[(n, verdict, quote, why)]; accepts the older {"unsupported": [...]} shape too."""
    raw = (raw or "").strip()
    raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw)
    try:
        data = json.loads(raw)
    except ValueError:
        return []
    if not isinstance(data, dict):
        return []
    found: List[Tuple[int, str, str, str]] = []
    for item in data.get("results") if isinstance(data.get("results"), list) else []:
        if isinstance(item, dict) and isinstance(item.get("n"), int) and 1 <= item["n"] <= n_claims:
            found.append((item["n"], str(item.get("verdict") or "").lower(), str(item.get("quote") or ""), str(item.get("why") or "")))
    for item in data.get("unsupported") if isinstance(data.get("unsupported"), list) else []:
        if isinstance(item, dict) and isinstance(item.get("n"), int) and 1 <= item["n"] <= n_claims:
            found.append((item["n"], "unsupported", "", str(item.get("why") or "no case document records this")))
    return found


def _passes() -> int:
    try:
        return max(1, min(4, int(os.getenv("LEGAL_ATTRIBUTION_PASSES", "2"))))
    except ValueError:
        return 2


def check_attributions(
    text: str,
    exhibits: List[str],
    judge: Callable[[str, str], str],
    max_claims: int = _MAX_CLAIMS,
    trace: Optional[list] = None,
) -> List[Tuple[str, str]]:
    """[(sentence, why)] for attributions no exhibit supports. Fails open: [] on any problem.

    The judge is a reasoning model and not deterministic: a probe missed the exact C1 sentence shape in
    2 of 3 calls. C1 is the critical failure, so recall matters more than saving one cheap call: the
    judge runs LEGAL_ATTRIBUTION_PASSES times (default 2) and a claim is reported if ANY pass flags it.
    A pass that errors is skipped; the code-side overrides apply to every pass. `trace`, when given,
    collects every verdict (for the probe and for debugging)."""
    if not exhibits or judge is None or not attribution_check_enabled():
        return []
    claims = _candidates(text, max_claims)
    if not claims:
        return []
    try:
        passages = _passages(exhibits)
        idf = _idf(passages)
        blocks: List[str] = []
        coverage: List[float] = []
        for n, (_sentence, claim, subject) in enumerate(claims, 1):
            chosen, cov = _excerpts(claim, subject, passages, idf)
            coverage.append(cov)
            lines = [f"CLAIM {n}: {claim}", "EXCERPTS:"]
            lines += [f"[{n}.{chr(96 + k)} | document {p.doc}: {p.header}] {p.text}" for k, p in enumerate(chosen, 1)]
            if not chosen:
                lines.append("(no excerpt of any case document is relevant to this claim)")
            blocks.append("\n".join(lines))
        prompt = "\n\n".join(blocks) + "\n\n" + _INSTRUCTIONS
        flagged: Dict[int, str] = {}
        for run in range(1, _passes() + 1):
            try:
                results = _parse(judge(_SYSTEM, prompt), len(claims))
            except Exception as e:  # one failed pass must not hide what the others found
                logging.warning("attribution check pass %d failed: %s", run, e)
                continue
            n_flagged = 0
            for n, verdict, quote, why in results:
                if trace is not None:
                    trace.append({"pass": run, "n": n, "claim": claims[n - 1][1], "verdict": verdict, "why": why, "quote": quote})
                if verdict != "unsupported":
                    continue
                if coverage[n - 1] >= _VERBATIM_COVERAGE:
                    continue  # near-verbatim in an exhibit: the judge is wrong, never act on it
                if quote.strip():
                    continue  # it quoted support while calling the claim unsupported: contradictory, so fail open
                n_flagged += 1
                flagged.setdefault(n, why or "no case document records this")
            logging.info("attribution check: pass %d judged %d claim(s), flagged %d", run, len(claims), n_flagged)
        return [(claims[n - 1][0], flagged[n]) for n in sorted(flagged)]
    except Exception as e:  # never let a failed check block or alter an answer
        logging.warning("attribution check failed (answer left as is): %s", e)
        return []
