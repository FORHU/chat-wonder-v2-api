"""Unit tests for sync_active_case_documents' cache-reuse-only grounding.

sync_active_case_documents used to eagerly call get_case_document (a blocking HTTP call to
ilovelawyer-api) once per attached document, every turn — for a case with many documents, that
serialized to minutes of latency and, before the asyncio.to_thread fix, froze the whole server
for every other session while it ran. It no longer fetches anything itself: it only reuses
documents already sitting in case_document_cache (populated by the model's own get_case_document
tool calls, via execute_function_call's cache-write side effect), still respecting the same
token budget those on-demand fetches are capped to.

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


def _seed_cache(state, doc_id: str, text: str, name: str = None):
    """Simulates a document the model already fetched itself earlier this session — the same
    entry shape execute_function_call's get_case_document side effect would have written."""
    entry = {"id": doc_id, "name": name or doc_id, "text": text, "chunk_count": 1}
    entry["_token_count"] = srv.calculate_tokens(text)
    state.case_document_cache[doc_id] = entry
    return entry


class NoEagerFetchTests(unittest.TestCase):
    def test_never_calls_get_case_document(self):
        """The whole point of the change: no network call happens just from syncing scope."""
        session_id = "test-no-eager-fetch"
        _make_session(session_id)

        with patch.object(srv, "get_case_document") as mocked:
            asyncio.run(srv.sync_active_case_documents(session_id, ["doc-a", "doc-b"], None))

        mocked.assert_not_called()

    def test_uncached_document_is_simply_absent(self):
        session_id = "test-uncached-absent"
        state = _make_session(session_id)

        asyncio.run(srv.sync_active_case_documents(session_id, ["doc-never-fetched"], None))

        self.assertEqual(state.active_case_documents, [])
        # Still known to the model via the manifest, even though not fetched into context.
        self.assertEqual(state.allowed_case_document_ids, {"doc-never-fetched"})


class CacheReuseTests(unittest.TestCase):
    def test_previously_cached_document_is_reused(self):
        session_id = "test-reuse"
        state = _make_session(session_id)
        _seed_cache(state, "doc-a", "actual content", name="Exhibit A")

        with patch.object(srv, "get_case_document") as mocked:
            asyncio.run(srv.sync_active_case_documents(session_id, ["doc-a"], None))
            mocked.assert_not_called()

        entry = next(d for d in state.active_case_documents if d["id"] == "doc-a")
        self.assertEqual(entry["text"], "actual content")

    def test_dropped_from_allowed_scope_is_evicted_from_cache(self):
        """A document cached under one case must not survive into a different case's turn —
        same guard as before, just re-verified against the cache-only rewrite."""
        session_id = "test-scope-drop"
        state = _make_session(session_id)
        _seed_cache(state, "doc-old-case", "stale content")

        asyncio.run(srv.sync_active_case_documents(session_id, ["doc-new-case"], None))

        self.assertNotIn("doc-old-case", state.case_document_cache)
        self.assertEqual(state.active_case_documents, [])


class TokenBudgetCapTests(unittest.TestCase):
    def test_more_than_ten_small_cached_documents_all_survive(self):
        session_id = "test-many-small"
        state = _make_session(session_id)
        doc_ids = [f"doc-{i}" for i in range(15)]
        for doc_id in doc_ids:
            _seed_cache(state, doc_id, "short excerpt")

        asyncio.run(srv.sync_active_case_documents(session_id, doc_ids, None))

        self.assertEqual(
            len(state.active_case_documents),
            15,
            "a flat document-count cap dropped documents even though total text was tiny",
        )

    def test_reused_cache_entries_still_respect_the_token_budget(self):
        """The budget cap used to be enforced only as part of the eager-fetch loop's eviction —
        confirms it's still applied when reconstructing active_case_documents from cache alone."""
        session_id = "test-over-budget"
        state = _make_session(session_id)
        # ~6000 tokens of filler each (well under 4 chars/token in practice, but comfortably
        # oversized either way) so a handful of these blow well past a 40K budget.
        big_text = "the quick brown fox jumps over the lazy dog. " * 1200
        doc_ids = [f"big-{i}" for i in range(10)]
        for doc_id in doc_ids:
            _seed_cache(state, doc_id, big_text)

        asyncio.run(srv.sync_active_case_documents(session_id, doc_ids, None))

        surviving_ids = [d["id"] for d in state.active_case_documents]
        self.assertNotIn("big-0", surviving_ids, "earliest-ordered large document should have been evicted")
        self.assertIn("big-9", surviving_ids, "last-ordered document should survive")
        total_tokens = sum(d.get("_token_count", 0) for d in state.active_case_documents)
        self.assertLessEqual(total_tokens, srv._CASE_DOCUMENT_TOKEN_BUDGET)


if __name__ == "__main__":
    unittest.main()
