"""Tests for get_case_document's BM25 reordering on the whole-document fallback path (see its
docstring in resources/functions/user_functions.py). No live ilovelawyer-api call —
_http_get_json is monkey-patched per test, same convention as test_case_document_grounding.py.
"""

import os
import unittest
from unittest.mock import patch

from resources.functions import user_functions as uf

# get_case_document reads these at call time.
os.environ.setdefault("ILOVELAWYER_API_BASE", "http://example.invalid")
os.environ.setdefault("CHAT_WONDER_API_KEY", "test-key")


def _chunk(chunk_id: str, index: int, text: str) -> dict:
    return {"id": chunk_id, "chunkIndex": index, "chunkText": text}


def _fake_response(chunks: list, name: str = "Exhibit.pdf", rag_status: str = "READY") -> dict:
    return {"name": name, "ragStatus": rag_status, "chunks": chunks}


class Bm25ReorderingTests(unittest.TestCase):
    def setUp(self):
        # A document whose combined text exceeds _CASE_DOCUMENT_CHAR_CAP (32000), padded so
        # only the reordering — not real content — decides what survives truncation. The
        # relevant chunk is last by chunkIndex, so natural order would drop it.
        filler = "irrelevant padding text about unrelated matters. " * 700  # ~35,700 chars
        self.chunks = [
            _chunk("c0", 0, filler),
            _chunk("c1", 1, "the notice of specified default under clause 1.7 was never served"),
        ]

    def test_relevant_chunk_survives_truncation_when_query_given(self):
        with patch.object(uf, "_http_get_json", return_value=_fake_response(self.chunks)):
            result = uf.get_case_document("doc1", query="clause 1.7 specified default")
        self.assertTrue(result["success"])
        self.assertIn("clause 1.7", result["text"])
        self.assertTrue(result["truncated"] if "truncated" in result else result["partial"])

    def test_relevant_chunk_lost_without_query_natural_order(self):
        """Control: same document, no query — old behaviour (chunkIndex order) drops c1."""
        with patch.object(uf, "_http_get_json", return_value=_fake_response(self.chunks)):
            result = uf.get_case_document("doc1")
        self.assertTrue(result["success"])
        self.assertNotIn("clause 1.7", result["text"])

    def test_document_under_cap_keeps_natural_chunk_index_order(self):
        small_chunks = [_chunk("a", 0, "first part"), _chunk("b", 1, "second part")]
        with patch.object(uf, "_http_get_json", return_value=_fake_response(small_chunks)):
            result = uf.get_case_document("doc1", query="second")
        self.assertEqual(result["text"], "first part\n\nsecond part")

    def test_case_document_chunk_ids_filter_bypasses_bm25_branch(self):
        """A pre-filtered read must stay exactly as before — no BM25 reordering applied."""
        with patch.object(uf, "_http_get_json", return_value=_fake_response(self.chunks)):
            result = uf.get_case_document("doc1", case_document_chunk_ids=["c0"], query="clause 1.7")
        self.assertNotIn("clause 1.7", result["text"])

    def test_explicit_paging_bypasses_bm25_branch(self):
        """limit implies sequential paging by chunkIndex — BM25 must not reorder mid-page."""
        with patch.object(uf, "_http_get_json", return_value=_fake_response(self.chunks)):
            result = uf.get_case_document("doc1", query="clause 1.7", limit=1)
        # Natural order: chunk 0 (the filler) comes first regardless of query relevance.
        self.assertNotIn("clause 1.7", result["text"])

    def test_no_query_no_reorder_even_over_cap(self):
        with patch.object(uf, "_http_get_json", return_value=_fake_response(self.chunks)):
            result = uf.get_case_document("doc1")
        self.assertTrue(result["text"].startswith("irrelevant padding"))


if __name__ == "__main__":
    unittest.main()
