"""Unit tests for Decision Records (differentiation program, Phase 1) — legal_decisions.py.

Pure verification logic only: anchor-in-answer matching, rule-link verification against the
retrieved pool (same rule as the Cite Gate), evidence doc-label resolution against the case
manifest, and quote verification against the case-document/search-result corpus. No live
OpenAI calls — the prompt-driving function (_generate_decision_records) lives in the_server.py
and is exercised end-to-end in tests/test_legal_verify.py-style fake-client tests, not here.
"""

import unittest

from legal_decisions import (
    audit_decision_records,
    case_document_labels,
    case_document_text_corpus,
    resolve_doc_label,
)

POOL = [
    {"id": "abc", "title": "Manalo v. Republic", "url": "https://juris.ph/case/abc"},
]

MANIFEST = [
    {"id": "doc-1", "name": "D01_HSE_Preliminary_Investigation_Report.pdf"},
    {"id": "doc-7", "name": "D07_Interview_Under_Caution_Aldwyn_Ferris.pdf"},
]

CASE_DOCS = [
    {"id": "doc-1", "name": "D01_HSE_Preliminary_Investigation_Report.pdf", "text": "The east tie line at levels 6 and 7 was found incomplete."},
    {"id": "doc-7", "name": "D07_Interview_Under_Caution_Aldwyn_Ferris.pdf", "text": "I was in Manchester at ten to ten."},
]

ANSWER = (
    "Meridian is likely liable for corporate manslaughter. "
    "The absence of the through-ties was a substantial cause of the collapse. "
    "Mr Ferris's account of his whereabouts is internally inconsistent."
)


def _record(**over):
    base = {
        "anchor": "The absence of the through-ties was a substantial cause of the collapse.",
        "conclusion": "The missing ties caused the collapse.",
        "rule": [{"title": "Corporate Manslaughter and Corporate Homicide Act 2007, s 1", "url": "https://juris.ph/case/abc"}],
        "evidenceFor": [{"doc": "D01", "pinpoint": "para 10", "quote": "east tie line at levels 6 and 7 was found incomplete"}],
        "evidenceAgainst": [],
        "alternatives": [{"position": "Wind alone caused the collapse", "whyRejected": "the bay would have survived a tied, compliant load", "evidenceRef": "D20.1"}],
        "weighting": "The physical survey outweighs disputed witness recollection.",
        "confidence": "high",
        "wouldChangeIf": ["The ties were found intact"],
    }
    base.update(over)
    return base


class DocLabelTests(unittest.TestCase):
    def test_short_code_resolves_to_full_manifest_name(self):
        labels = case_document_labels(CASE_DOCS, MANIFEST)
        self.assertEqual(resolve_doc_label("D01", labels), "doc-1")
        self.assertEqual(resolve_doc_label("d07", labels), "doc-7")

    def test_full_name_also_resolves(self):
        labels = case_document_labels(CASE_DOCS, MANIFEST)
        self.assertEqual(resolve_doc_label("D01_HSE_Preliminary_Investigation_Report.pdf", labels), "doc-1")

    def test_unknown_label_does_not_resolve(self):
        labels = case_document_labels(CASE_DOCS, MANIFEST)
        self.assertIsNone(resolve_doc_label("D99", labels))
        self.assertIsNone(resolve_doc_label("", labels))

    def test_falls_back_to_case_documents_without_manifest(self):
        labels = case_document_labels(CASE_DOCS, None)
        self.assertEqual(resolve_doc_label("D01", labels), "doc-1")


class CorpusTests(unittest.TestCase):
    def test_case_document_corpus_is_normalized_and_searchable(self):
        corpus = case_document_text_corpus(CASE_DOCS)
        self.assertIn("east tie line at levels 6 and 7 was found incomplete", corpus)

    def test_empty_case_documents_give_empty_corpus(self):
        self.assertEqual(case_document_text_corpus(None), "")
        self.assertEqual(case_document_text_corpus([]), "")


class AuditDecisionRecordsTests(unittest.TestCase):
    def test_well_formed_record_is_fully_verified(self):
        records, stats = audit_decision_records([_record()], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r["rule"], [{"title": "Corporate Manslaughter and Corporate Homicide Act 2007, s 1", "url": "https://juris.ph/case/abc", "verified": True}])
        self.assertTrue(r["evidenceFor"][0]["verified"])
        self.assertEqual(r["evidenceFor"][0]["docId"], "doc-1")
        self.assertEqual(stats.records_in, 1)
        self.assertEqual(stats.records_out, 1)
        self.assertEqual(stats.anchors_unverified, 0)
        self.assertEqual(stats.rules_verified, 1)
        self.assertEqual(stats.rules_dropped, 0)

    def test_record_with_anchor_not_in_answer_is_dropped(self):
        rec = _record(anchor="This sentence was never in the answer.")
        records, stats = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(records, [])
        self.assertEqual(stats.anchors_unverified, 1)
        self.assertEqual(stats.records_out, 0)

    def test_anchor_matching_tolerates_whitespace_differences(self):
        rec = _record(anchor="The absence of the through-ties   was a substantial cause of the collapse.")
        records, stats = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(len(records), 1)
        self.assertEqual(stats.anchors_unverified, 0)

    def test_rule_link_not_in_pool_is_dropped_to_bare_title(self):
        rec = _record(rule=[{"title": "Invented Case", "url": "https://juris.ph/case/does-not-exist"}])
        records, stats = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(records[0]["rule"], [{"title": "Invented Case", "url": None, "verified": False}])
        self.assertEqual(stats.rules_dropped, 1)
        self.assertEqual(stats.rules_verified, 0)

    def test_rule_with_no_url_is_kept_as_unverified_plain_title(self):
        rec = _record(rule=[{"title": "General principle, no specific authority"}])
        records, _ = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(records[0]["rule"], [{"title": "General principle, no specific authority", "url": None, "verified": False}])

    def test_evidence_doc_not_attached_is_unverified_but_kept(self):
        rec = _record(evidenceFor=[{"doc": "D99", "pinpoint": "para 1", "quote": "something"}])
        records, stats = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        ev = records[0]["evidenceFor"][0]
        self.assertIsNone(ev["docId"])
        self.assertFalse(ev["verified"])
        self.assertEqual(stats.evidence_unverified, 1)

    def test_evidence_quote_not_in_document_is_unverified(self):
        rec = _record(evidenceFor=[{"doc": "D01", "pinpoint": "para 10", "quote": "a quote that does not appear anywhere in the exhibits"}])
        records, stats = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        ev = records[0]["evidenceFor"][0]
        self.assertEqual(ev["docId"], "doc-1")  # doc resolves fine
        self.assertFalse(ev["verified"])  # but the quote doesn't check out
        self.assertEqual(stats.quotes_unverified, 1)

    def test_evidence_without_quote_only_needs_doc_resolution(self):
        rec = _record(evidenceFor=[{"doc": "D07", "pinpoint": "para 8"}])
        records, stats = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        ev = records[0]["evidenceFor"][0]
        self.assertTrue(ev["verified"])
        self.assertEqual(stats.quotes_verified, 0)
        self.assertEqual(stats.quotes_unverified, 0)

    def test_evidence_against_audited_same_as_evidence_for(self):
        rec = _record(evidenceAgainst=[{"doc": "D99", "pinpoint": "x"}])
        records, stats = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertFalse(records[0]["evidenceAgainst"][0]["verified"])
        self.assertEqual(stats.evidence_unverified, 1)

    def test_confidence_defaults_to_medium_when_invalid(self):
        rec = _record(confidence="extremely high")
        records, _ = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(records[0]["confidence"], "medium")

    def test_alternatives_missing_both_fields_are_dropped(self):
        rec = _record(alternatives=[{"position": "", "whyRejected": ""}, {"position": "Real alt", "whyRejected": "because"}])
        records, _ = audit_decision_records([rec], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(len(records[0]["alternatives"]), 1)
        self.assertEqual(records[0]["alternatives"][0]["position"], "Real alt")

    def test_non_list_input_returns_empty(self):
        records, stats = audit_decision_records(None, ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(records, [])
        self.assertEqual(stats.records_in, 0)
        records, _ = audit_decision_records("not a list", ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(records, [])

    def test_malformed_entries_in_list_are_skipped_not_fatal(self):
        records, stats = audit_decision_records([_record(), "garbage", 42, None], ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertEqual(len(records), 1)
        self.assertEqual(stats.records_in, 4)

    def test_caps_at_max_records(self):
        many = [_record() for _ in range(20)]
        records, stats = audit_decision_records(many, ANSWER, POOL, CASE_DOCS, MANIFEST)
        self.assertLessEqual(len(records), 8)


if __name__ == "__main__":
    unittest.main()
