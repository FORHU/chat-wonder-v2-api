"""Unit tests for sync_active_case_documents' document-count-independent grounding.

Covers two fixes made alongside ilovelawyer-api's per-document chunk floor:
  - a document attached to the case but with no chunks sampled this turn used to vanish with
    zero trace (execute_function_call's return value was discarded) instead of surfacing the
    honest "no chunks relevant" message get_case_document already produces;
  - active_case_documents used to hard-cap at the last 10 fetched documents regardless of size,
    discarding whole documents past the 10th even when their content was tiny; it's now a token
    budget, so many small documents fit and a few huge ones still get capped sensibly.

No live OpenAI/juris.ph/ilovelawyer-api calls — get_case_document is monkey-patched per test.
"""

import asyncio
import unittest
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)


def _make_session(session_id: str) -> "srv.ChatState":
    state = srv.ChatState()
    srv._context.sessions[session_id] = state
    return state


class NoChunksSampledTests(unittest.TestCase):
    def test_document_with_zero_sampled_chunks_gets_an_honest_stand_in(self):
        session_id = "test-no-chunks"
        state = _make_session(session_id)

        def fake_get_case_document(case_document_id, case_document_chunk_ids=None):
            return {
                "success": True,
                "id": case_document_id,
                "name": "Exhibit D",
                "chunks_found": 0,
                "message": "No chunks relevant to the current question were found in this document.",
            }

        with patch.object(srv, "get_case_document", fake_get_case_document):
            asyncio.run(srv.sync_active_case_documents(session_id, ["doc-d"], ["some-other-chunk-id"]))

        ids = [d["id"] for d in state.active_case_documents]
        self.assertIn("doc-d", ids, "document silently vanished instead of getting a stand-in")
        entry = next(d for d in state.active_case_documents if d["id"] == "doc-d")
        self.assertIn("No chunks relevant", entry["text"])

    def test_document_with_real_text_is_unaffected(self):
        session_id = "test-real-text"
        state = _make_session(session_id)

        def fake_get_case_document(case_document_id, case_document_chunk_ids=None):
            return {"success": True, "id": case_document_id, "name": "Exhibit A", "text": "actual content", "chunk_count": 3}

        with patch.object(srv, "get_case_document", fake_get_case_document):
            asyncio.run(srv.sync_active_case_documents(session_id, ["doc-a"], None))

        entry = next(d for d in state.active_case_documents if d["id"] == "doc-a")
        self.assertEqual(entry["text"], "actual content")


class TokenBudgetCapTests(unittest.TestCase):
    def test_more_than_ten_small_documents_all_survive(self):
        session_id = "test-many-small"
        state = _make_session(session_id)
        doc_ids = [f"doc-{i}" for i in range(15)]

        def fake_get_case_document(case_document_id, case_document_chunk_ids=None):
            return {"success": True, "id": case_document_id, "name": case_document_id, "text": "short excerpt", "chunk_count": 1}

        with patch.object(srv, "get_case_document", fake_get_case_document):
            asyncio.run(srv.sync_active_case_documents(session_id, doc_ids, None))

        self.assertEqual(
            len(state.active_case_documents),
            15,
            "a flat document-count cap dropped documents even though total text was tiny",
        )

    def test_oldest_document_evicted_first_once_over_budget(self):
        session_id = "test-over-budget"
        state = _make_session(session_id)
        # ~6000 tokens of filler each (well under 4 chars/token in practice, but comfortably
        # oversized either way) so a handful of these blow well past a 40K budget.
        big_text = "the quick brown fox jumps over the lazy dog. " * 1200
        doc_ids = [f"big-{i}" for i in range(10)]

        def fake_get_case_document(case_document_id, case_document_chunk_ids=None):
            return {"success": True, "id": case_document_id, "name": case_document_id, "text": big_text, "chunk_count": 50}

        with patch.object(srv, "get_case_document", fake_get_case_document):
            asyncio.run(srv.sync_active_case_documents(session_id, doc_ids, None))

        surviving_ids = [d["id"] for d in state.active_case_documents]
        self.assertNotIn("big-0", surviving_ids, "oldest large document should have been evicted")
        self.assertIn("big-9", surviving_ids, "most recently processed document should survive")
        total_tokens = sum(srv.calculate_tokens(d["text"]) for d in state.active_case_documents)
        self.assertLessEqual(total_tokens, srv._CASE_DOCUMENT_TOKEN_BUDGET)


if __name__ == "__main__":
    unittest.main()
