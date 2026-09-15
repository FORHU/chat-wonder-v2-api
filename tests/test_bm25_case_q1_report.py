"""Q1 case study: can get_case_document's BM25 truncation ranking (bm25.py, wired in at
resources/functions/user_functions.py::get_case_document) be trusted on its own to surface the
right chunk out of a real-shaped case bundle?

This matters now specifically because ilovelawyer-api's /case-document endpoints stopped
returning each chunk's embedding vector (see chat-wonder-v2-api's sibling repo,
document-chunk.repository.ts::findByDocument) -- BM25 is lexical-only (term frequency / inverse
document frequency, no semantic similarity), so get_case_document's whole-document truncation
fallback has no embedding signal to fall back on if BM25 picks wrong. This is the one case study
requested ("Q1") rather than a full benchmark sweep -- one realistic query against one realistic
case bundle, run for real (no live ilovelawyer-api/OpenAI calls -- _http_get_json is
monkey-patched, same convention as test_case_document_bm25_ordering.py), with the actual BM25
scores printed so the ranking can be inspected, not just asserted on faith.

Run directly for the printed report:
    python -m tests.test_bm25_case_q1_report
Run as a regression check:
    python -m unittest tests.test_bm25_case_q1_report
"""

import os
import unittest
from unittest.mock import patch

from resources.functions import user_functions as uf
from bm25 import BM25

os.environ.setdefault("ILOVELAWYER_API_BASE", "http://example.invalid")
os.environ.setdefault("CHAT_WONDER_API_KEY", "test-key")

# --- Q1: the case bundle -----------------------------------------------------------------
# "Whitfield v Carrow Logistics Ltd" (fictional UK Employment Tribunal matter). Three chunks
# carry real content; three are bulk filler so the combined text exceeds _CASE_DOCUMENT_CHAR_CAP
# (32000) and the BM25 truncation path actually triggers -- mirrors the shape of
# test_case_document_bm25_ordering.py. All three real chunks are placed *before* the filler by
# chunkIndex is deliberately avoided: they sit interleaved/late so natural chunkIndex order would
# not obviously favor them, matching how a real exhibit's relevant paragraph can land anywhere.

_FILLER = "This section records routine case-management correspondence of no substantive relevance. " * 130  # ~11,830 chars

# GOLD: actually answers Q1, but uses "Judgment" (the correct legal term for a Tribunal's ruling)
# instead of the query's word "decision" -- a realistic phrasing mismatch, not a contrived one.
_GOLD = (
    "Any party wishing to appeal against this Judgment must present a notice of appeal to the "
    "Employment Appeal Tribunal within 42 days of the date on which the written reasons were "
    "sent to the parties."
)

# DECOY 1: an internal grievance policy. Shares "appeal" and "decision" (literally, unlike GOLD)
# and "days", but the deadline it states is for a different, unrelated procedure.
_DECOY_GRIEVANCE = (
    "An employee who wishes to appeal a decision made under this grievance procedure must "
    "submit their appeal in writing within 5 working days of receiving notification of the "
    "decision."
)

# DECOY 2: a case-management order. Shares "file", "Tribunal", "deadline" and "days", but is
# about the witness-statement deadline, not the appeal deadline.
_DECOY_CASE_MANAGEMENT = (
    "The parties must file their witness statements and any supporting documents with the "
    "Tribunal no later than 21 days before the final hearing. Failure to file by this deadline "
    "may result in the evidence being excluded."
)

_CHUNKS = [
    {"id": "filler-0", "chunkIndex": 0, "chunkText": _FILLER},
    {"id": "grievance-decoy", "chunkIndex": 1, "chunkText": _DECOY_GRIEVANCE},
    {"id": "filler-1", "chunkIndex": 2, "chunkText": _FILLER},
    {"id": "case-mgmt-decoy", "chunkIndex": 3, "chunkText": _DECOY_CASE_MANAGEMENT},
    {"id": "filler-2", "chunkIndex": 4, "chunkText": _FILLER},
    {"id": "gold-appeal-deadline", "chunkIndex": 5, "chunkText": _GOLD},
]

_Q1 = "What is the deadline for the Claimant to file an appeal against the Tribunal's decision?"

_GOLD_ID = "gold-appeal-deadline"


def _fake_response(chunks):
    return {"name": "Whitfield v Carrow Logistics Ltd - bundle.pdf", "ragStatus": "READY", "chunks": chunks}


class Bm25Q1CaseStudy(unittest.TestCase):
    def setUp(self):
        total_chars = sum(len(c["chunkText"]) for c in _CHUNKS)
        # Sanity-check the fixture itself exercises the path under test (see get_case_document's
        # docstring: BM25 reordering only fires when combined chunk text exceeds the cap).
        self.assertGreater(total_chars, uf._CASE_DOCUMENT_CHAR_CAP, "fixture must exceed _CASE_DOCUMENT_CHAR_CAP to exercise the BM25 branch")

    def test_bm25_scores_and_get_case_document_report(self):
        texts = [c["chunkText"] for c in _CHUNKS]
        ids = [c["id"] for c in _CHUNKS]

        bm25 = BM25(texts)
        raw_scores = bm25.scores(_Q1)
        order = sorted(range(len(raw_scores)), key=lambda i: raw_scores[i], reverse=True)

        with patch.object(uf, "_http_get_json", return_value=_fake_response(_CHUNKS)):
            result = uf.get_case_document("whitfield-bundle", query=_Q1)

        gold_rank = order.index(ids.index(_GOLD_ID)) + 1  # 1-based
        gold_survived = result["success"] and _GOLD_ID and ("42 days" in result.get("text", ""))

        report_lines = [
            "",
            "=" * 78,
            "Q1 CASE STUDY REPORT -- get_case_document BM25 truncation ranking",
            "=" * 78,
            f"Case bundle: {_fake_response(_CHUNKS)['name']}",
            f"Query (Q1): {_Q1!r}",
            f"Total bundle chars: {sum(len(t) for t in texts)} (cap: {uf._CASE_DOCUMENT_CHAR_CAP})",
            "",
            f"{'rank':>4}  {'chunk id':<20}  {'bm25 score':>10}  content",
            "-" * 78,
        ]
        for rank, idx in enumerate(order, start=1):
            snippet = texts[idx][:70].replace("\n", " ")
            marker = "  <-- GOLD" if ids[idx] == _GOLD_ID else ""
            report_lines.append(f"{rank:>4}  {ids[idx]:<20}  {raw_scores[idx]:>10.4f}  {snippet}...{marker}")
        report_lines += [
            "",
            f"get_case_document(query=Q1) -> success={result['success']}, partial={result.get('partial')}, "
            f"truncated={'[...truncated...]' in result.get('text', '')}",
            f"GOLD chunk's BM25 rank: {gold_rank} of {len(_CHUNKS)}",
            f"GOLD fact ('42 days') present in returned text: {gold_survived}",
            "",
            "ANALYSIS:",
        ]
        if gold_survived and gold_rank == 1:
            report_lines.append(
                "  PASS -- BM25 ranked the gold chunk first despite it using 'Judgment' rather than the "
                "query's 'decision', and despite two decoys sharing more literal query terms "
                "('decision', 'file', 'deadline'). IDF weighting suppressed the decoys' overlap because "
                "those terms are common across the bundle, while 'appeal'/'Tribunal' co-occurring with "
                "'42 days' carried more weight."
            )
        elif gold_survived:
            report_lines.append(
                f"  PARTIAL PASS -- the gold chunk survived truncation (rank {gold_rank}) and its answer "
                "made it into context, but it was not BM25's top pick. On a tighter char budget (more "
                "filler, or a lower cap) it could have been truncated away."
            )
        else:
            report_lines.append(
                "  FAIL -- pure lexical BM25 ranked a decoy above the chunk that actually answers Q1, and "
                "the answer did not survive truncation. This is the expected failure mode of a term-overlap "
                "ranker: a decoy chunk that happens to repeat the query's literal words ('decision', 'file', "
                "'deadline') outranks a chunk that answers the question using different but synonymous "
                "legal phrasing ('Judgment'). Since the case-document endpoint no longer returns "
                "embeddings, there is currently no semantic signal to fall back on here -- "
                "bm25.py's own rrf_rank_scores (BM25 + embedding fusion) exists but is not wired into "
                "this path, and even if it were, this endpoint would need to expose a similarity score "
                "again (not necessarily the raw vector) to feed it."
            )
        report_lines.append("=" * 78)
        print("\n".join(report_lines))

        # Structural assertions -- the mechanics under test, not a bet on ranking quality:
        self.assertTrue(result["success"])
        self.assertGreater(len(result.get("text", "")), 0)
        # The BM25 branch must actually have fired for this fixture (query given, no explicit
        # chunk id filter/limit, bundle over cap) -- if this ever goes False the report above is
        # measuring the wrong code path (natural chunkIndex order), not BM25 ranking quality.
        bm25_branch_fired = result.get("partial") in (True,) or "[...truncated...]" in result.get("text", "")
        self.assertTrue(bm25_branch_fired, "fixture did not exercise truncation -- adjust filler size")


if __name__ == "__main__":
    unittest.main()
