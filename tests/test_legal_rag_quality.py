"""Focused tests for legal RAG ingestion dedupe/quality and retrieval snippet selection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock

# Allow `python tests/test_legal_rag_quality.py` from repo root without install.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.ingestion import (  # noqa: E402
    IngestionStats,
    LegalCorpusIngestor,
    build_document_source_hash,
    extract_record_year,
    extract_stable_identity,
    extract_year_from_value,
    is_invalid_date_value,
    is_invalid_year,
)
from legal_rag.retrieval import (  # noqa: E402
    aggregate_merged_hits,
    select_best_chunk_row,
)


def _ingestor() -> LegalCorpusIngestor:
    return LegalCorpusIngestor(db=MagicMock(), s3=MagicMock(), embeddings=MagicMock(), s3_prefix="")


def _long_text(prefix: str = "Opinion body") -> str:
    return (prefix + " ") * 20  # comfortably above MIN_INDEX_CHARS


class TestRecordKeyPriority(unittest.TestCase):
    def test_same_case_no_different_uuid_remain_separate(self):
        ing = _ingestor()
        a = {"case_no": "CTA-001", "uuid": "aaa-111", "url": "https://example.com/a"}
        b = {"case_no": "CTA-001", "uuid": "bbb-222", "url": "https://example.com/b"}
        self.assertNotEqual(ing._record_key(a, "fb"), ing._record_key(b, "fb"))

    def test_same_case_no_different_url_remain_separate_without_uuid(self):
        ing = _ingestor()
        a = {"case_no": "BOC-RICE", "url": "https://example.com/shipments/1"}
        b = {"case_no": "BOC-RICE", "url": "https://example.com/shipments/2"}
        self.assertNotEqual(ing._record_key(a, "fb"), ing._record_key(b, "fb"))

    def test_exact_duplicates_with_same_stable_id_merge(self):
        ing = _ingestor()
        a = {"uuid": "same-uuid", "case_no": "X-1", "url": "https://a.example/1"}
        b = {"uuid": "same-uuid", "case_no": "X-1", "url": "https://b.example/2"}
        self.assertEqual(ing._record_key(a, "fb"), ing._record_key(b, "fb"))

    def test_uuid_preferred_over_case_no(self):
        ing = _ingestor()
        key = ing._record_key({"case_no": "CASE-9", "uuid": "u-9"}, "fb")
        self.assertEqual(key, "u-9")

    def test_url_preferred_over_case_no(self):
        ing = _ingestor()
        key = ing._record_key({"case_no": "CASE-9", "url": "https://example.com/doc"}, "fb")
        self.assertEqual(key, "https://example.com/doc")


class TestMergeRecords(unittest.TestCase):
    def test_does_not_replace_good_values_with_empty(self):
        ing = _ingestor()
        old = {
            "title": "Shipments of Rice",
            "full_text": _long_text("Full opinion"),
            "summary": "A useful summary about rice shipments and customs duties.",
            "uuid": "keep-me",
            "source_url": "https://example.com/good",
        }
        new = {
            "title": "",
            "full_text": None,
            "summary": "   ",
            "uuid": "",
            "source_url": None,
            "case_no": "BOC-1",
        }
        merged = ing._merge_records(old, new, prefer_new=True)
        self.assertEqual(merged["title"], "Shipments of Rice")
        self.assertEqual(merged["full_text"], old["full_text"])
        self.assertEqual(merged["summary"], old["summary"])
        self.assertEqual(merged["uuid"], "keep-me")
        self.assertEqual(merged["source_url"], "https://example.com/good")
        self.assertEqual(merged["case_no"], "BOC-1")

    def test_prefer_new_keeps_better_non_empty_full_details(self):
        ing = _ingestor()
        old = {"title": "List title", "summary": "Short list summary that is still long enough."}
        new = {
            "title": "Full details title",
            "full_text": _long_text("Detailed"),
            "summary": "Richer full-details summary for the same matter.",
        }
        merged = ing._merge_records(old, new, prefer_new=True)
        self.assertEqual(merged["title"], "Full details title")
        self.assertEqual(merged["full_text"], new["full_text"])
        self.assertEqual(merged["summary"], new["summary"])

    def test_keeps_full_text_when_other_side_lacks_it(self):
        ing = _ingestor()
        list_row = {"title": "T", "summary": _long_text("Summary")}
        detail_row = {"title": "T", "full_text": _long_text("Full")}
        merged = ing._merge_records(list_row, detail_row, prefer_new=True)
        self.assertEqual(merged["full_text"], detail_row["full_text"])
        merged_rev = ing._merge_records(detail_row, {"title": "T", "full_text": ""}, prefer_new=True)
        self.assertEqual(merged_rev["full_text"], detail_row["full_text"])


class TestYearQualityGuards(unittest.TestCase):
    def test_1970_is_invalid_year(self):
        self.assertTrue(is_invalid_year(1970, reference_year=2026))
        self.assertTrue(is_invalid_date_value("1970-01-01", reference_year=2026))
        self.assertIsNone(extract_year_from_value(1970, reference_year=2026))
        self.assertIsNone(extract_year_from_value("1970-01-01T00:00:00Z", reference_year=2026))

    def test_valid_year_accepted(self):
        self.assertFalse(is_invalid_year(2019, reference_year=2026))
        self.assertEqual(extract_year_from_value(2019, reference_year=2026), 2019)
        self.assertEqual(extract_year_from_value("2019-06-15", reference_year=2026), 2019)

    def test_future_and_malformed_years_rejected(self):
        self.assertIsNone(extract_year_from_value(2099, reference_year=2026))
        self.assertIsNone(extract_year_from_value("not-a-date", reference_year=2026))
        self.assertIsNone(extract_year_from_value("", reference_year=2026))

    def test_record_year_nulls_out_1970_but_preserves_usable_date(self):
        self.assertIsNone(extract_record_year({"year": 1970, "date": "1970-01-01"}, reference_year=2026))
        self.assertEqual(
            extract_record_year({"year": 1970, "date": "2015-03-01"}, reference_year=2026),
            2015,
        )


class TestSourceHashStableIdentity(unittest.TestCase):
    """Regression: source_hash must use stable IDs, not only case_no|title|year|path."""

    _BASE = {
        "case_no": "CTA-001",
        "title": "Shipments of Rice",
        "year": 2019,
    }
    _PATH = "json/court_of_tax_appeals/cta/anycase_cta.json"
    _CAT = "court_of_tax_appeals"
    _BUCKET = "cta"

    def test_different_uuid_same_case_title_year_different_source_hash(self):
        a = {**self._BASE, "uuid": "aaa-111"}
        b = {**self._BASE, "uuid": "bbb-222"}
        ha = build_document_source_hash(
            a, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path=self._PATH
        )
        hb = build_document_source_hash(
            b, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path=self._PATH
        )
        self.assertNotEqual(ha, hb)
        self.assertTrue(extract_stable_identity(a).startswith("uuid:"))
        self.assertNotEqual(extract_stable_identity(a), extract_stable_identity(b))

    def test_different_url_same_case_title_year_different_source_hash(self):
        a = {**self._BASE, "url": "https://example.com/docs/1"}
        b = {**self._BASE, "url": "https://example.com/docs/2"}
        ha = build_document_source_hash(
            a, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path=self._PATH
        )
        hb = build_document_source_hash(
            b, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path=self._PATH
        )
        self.assertNotEqual(ha, hb)
        self.assertNotEqual(
            extract_stable_identity(a, s3_json_path=self._PATH),
            extract_stable_identity(b, s3_json_path=self._PATH),
        )

    def test_same_stable_identifier_same_source_hash(self):
        a = {**self._BASE, "uuid": "same-uuid", "url": "https://a.example/1"}
        b = {**self._BASE, "uuid": "same-uuid", "url": "https://b.example/2"}
        ha = build_document_source_hash(
            a, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path=self._PATH
        )
        hb = build_document_source_hash(
            b, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path="other/path.json"
        )
        self.assertEqual(ha, hb)
        self.assertEqual(extract_stable_identity(a), extract_stable_identity(b))

    def test_fallback_identity_without_stable_identifier(self):
        sparse = {**self._BASE}
        identity = extract_stable_identity(sparse, s3_json_path=self._PATH)
        self.assertTrue(identity.startswith("case_title_year:"))
        self.assertIn("cta-001", identity)
        hash_a = build_document_source_hash(
            sparse, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path=self._PATH
        )
        hash_b = build_document_source_hash(
            sparse, category=self._CAT, bucket_slug=self._BUCKET, s3_json_path=self._PATH
        )
        self.assertEqual(hash_a, hash_b)
        # Different s3 path keeps sparse rows distinct (no regression vs old hash).
        hash_other_path = build_document_source_hash(
            sparse,
            category=self._CAT,
            bucket_slug=self._BUCKET,
            s3_json_path="json/other/file.json",
        )
        self.assertNotEqual(hash_a, hash_other_path)

    def test_upsert_persists_distinct_source_hash_for_different_uuids(self):
        ing = _ingestor()
        ing.db.get_document_hashes.return_value = None
        ing.db.upsert_document.return_value = 1
        ing.db.replace_document_chunks.return_value = 1
        ing.embeddings.embed_texts.return_value = [[0.1, 0.2]]

        shared = {
            **self._BASE,
            "summary": _long_text("CTA opinion summary text"),
            "_s3_path": self._PATH,
        }
        hashes = []
        for uuid in ("uuid-one", "uuid-two"):
            stats = IngestionStats()
            ing._upsert_record(
                {**shared, "uuid": uuid},
                self._CAT,
                self._BUCKET,
                "manifests/x.manifest.json",
                stats,
            )
            hashes.append(ing.db.upsert_document.call_args[0][0]["source_hash"])

        self.assertEqual(len(hashes), 2)
        self.assertNotEqual(hashes[0], hashes[1])


class TestUpsertQualityGuards(unittest.TestCase):
    def test_skips_short_index_text(self):
        ing = _ingestor()
        stats = IngestionStats()
        ing._upsert_record(
            {"title": "T", "case_no": "1", "summary": "too short"},
            "cat",
            "bucket",
            "manifests/x.manifest.json",
            stats,
        )
        self.assertEqual(stats.skipped_empty, 1)
        ing.db.upsert_document.assert_not_called()

    def test_skips_missing_title_and_case_no(self):
        ing = _ingestor()
        stats = IngestionStats()
        ing._upsert_record(
            {"summary": _long_text("Useful text without identifiers")},
            "cat",
            "bucket",
            "manifests/x.manifest.json",
            stats,
        )
        self.assertEqual(stats.skipped_empty, 1)
        ing.db.upsert_document.assert_not_called()

    def test_keeps_doc_but_nulls_dirty_1970_year(self):
        ing = _ingestor()
        ing.db.get_document_hashes.return_value = None
        ing.db.upsert_document.return_value = 42
        ing.db.replace_document_chunks.return_value = 1
        ing.embeddings.embed_texts.return_value = [[0.1, 0.2]]
        stats = IngestionStats()
        record = {
            "title": "Shipments of Rice",
            "case_no": "BOC-1",
            "year": 1970,
            "date": "1970-01-01",
            "summary": _long_text("Customs ruling summary"),
            "url": "https://example.com/rice/1",
        }
        ing._upsert_record(record, "cat", "bucket", "manifests/x.manifest.json", stats)
        self.assertEqual(stats.documents_upserted, 1)
        payload = ing.db.upsert_document.call_args[0][0]
        self.assertIsNone(payload["year"])
        # Raw dirty metadata preserved for audit.
        self.assertEqual(payload["metadata_json"]["year"], 1970)


class TestRetrievalSnippetSelection(unittest.TestCase):
    def test_select_best_chunk_not_lexicographic_max(self):
        rows = [
            {"id": 1, "snippet": "zzz lexicographically last", "keyword_score": 0.1, "vector_score": 0.0},
            {"id": 1, "snippet": "aaa best vector hit", "keyword_score": 0.0, "vector_score": 0.95},
            {"id": 1, "snippet": "mmm mid text", "keyword_score": 0.5, "vector_score": 0.1},
        ]
        best = select_best_chunk_row(rows)
        self.assertEqual(best["snippet"], "aaa best vector hit")
        # Lexicographic MAX(snippet) would wrongly pick the zzz row.
        self.assertNotEqual(max(r["snippet"] for r in rows), best["snippet"])

    def test_aggregate_uses_top_ranked_chunk_snippet(self):
        rows = [
            {
                "id": 7,
                "title": "Doc",
                "case_no": "C-1",
                "bucket_slug": "b",
                "category": "c",
                "year": 2020,
                "source_url": None,
                "s3_json_path": "p",
                "s3_manifest_path": "m",
                "summary": "s",
                "snippet": "zzz weak keyword-only chunk",
                "keyword_score": 0.9,
                "vector_score": 0.0,
            },
            {
                "id": 7,
                "title": "Doc",
                "case_no": "C-1",
                "bucket_slug": "b",
                "category": "c",
                "year": 2020,
                "source_url": None,
                "s3_json_path": "p",
                "s3_manifest_path": "m",
                "summary": "s",
                "snippet": "aaa strongest hybrid chunk",
                "keyword_score": 0.2,
                "vector_score": 0.99,
            },
        ]
        out = aggregate_merged_hits(rows, limit=5)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["snippet"], "aaa strongest hybrid chunk")
        self.assertAlmostEqual(out[0]["keyword_score"], 0.9)
        self.assertAlmostEqual(out[0]["vector_score"], 0.99)


if __name__ == "__main__":
    unittest.main()
