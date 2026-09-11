"""Unit tests for canonical legislation.gov.uk URL enrichment (uk_legal_mcp/urls.py)."""

import unittest

from uk_legal_mcp.urls import (
    enrich_resolve_result,
    enrich_section_result,
    find_act_identity,
    section_url,
)

SEARCH_ROW = {
    "results": [
        {"title": "Employment Rights Act 1996", "type": "ukpga", "year": 1996, "number": 18, "url": "https://www.legislation.gov.uk/ukpga/1996/18"},
    ]
}


class SectionUrlTests(unittest.TestCase):
    def test_builds_canonical_url(self):
        self.assertEqual(section_url("ukpga", 1996, 18, "94"), "https://www.legislation.gov.uk/ukpga/1996/18/section/94")

    def test_missing_parts_return_none(self):
        self.assertIsNone(section_url("ukpga", None, 18, "94"))
        self.assertIsNone(section_url("", 1996, 18, "94"))
        self.assertIsNone(section_url("ukpga", 1996, 18, ""))

    def test_enrich_section_result_adds_url_from_args(self):
        result = {"success": True, "title": "The right.", "section_number": "94", "content": "..."}
        enrich_section_result(result, {"type": "ukpga", "year": 1996, "number": 18, "section": "94"})
        self.assertEqual(result["url"], "https://www.legislation.gov.uk/ukpga/1996/18/section/94")

    def test_enrich_section_result_keeps_existing_url_and_skips_failures(self):
        keep = {"success": True, "url": "https://example/keep"}
        enrich_section_result(keep, {"type": "ukpga", "year": 1996, "number": 18, "section": "94"})
        self.assertEqual(keep["url"], "https://example/keep")
        failed = {"success": False, "error": "nope"}
        enrich_section_result(failed, {"type": "ukpga", "year": 1996, "number": 18, "section": "94"})
        self.assertNotIn("url", failed)


class ResolveUrlTests(unittest.TestCase):
    def test_find_act_identity_from_search_row(self):
        self.assertEqual(find_act_identity([SEARCH_ROW], "Employment Rights Act 1996"), {"type": "ukpga", "year": 1996, "number": 18})

    def test_find_act_identity_from_act_url_only(self):
        pool = [{"title": "Housing Act 2004", "url": "https://www.legislation.gov.uk/ukpga/2004/34"}]
        self.assertEqual(find_act_identity(pool, "housing act 2004"), {"type": "ukpga", "year": "2004", "number": "34"})

    def test_search_page_resolved_url_rewritten_to_section(self):
        result = {
            "success": True, "type": "legislation", "legislation_title": "Employment Rights Act 1996",
            "section": "94", "resolved_url": "https://www.legislation.gov.uk/search?title=Employment+Rights+Act+1996",
        }
        enrich_resolve_result(result, [SEARCH_ROW])
        self.assertEqual(result["resolved_url"], "https://www.legislation.gov.uk/ukpga/1996/18/section/94")
        self.assertEqual(result["search_url"], "https://www.legislation.gov.uk/search?title=Employment+Rights+Act+1996")

    def test_search_page_without_section_becomes_act_url(self):
        result = {"success": True, "type": "legislation", "legislation_title": "Employment Rights Act 1996", "section": None, "resolved_url": "https://www.legislation.gov.uk/search?title=X"}
        enrich_resolve_result(result, [SEARCH_ROW])
        self.assertEqual(result["resolved_url"], "https://www.legislation.gov.uk/ukpga/1996/18")

    def test_untouched_when_identity_unknown_or_not_search_url(self):
        unknown = {"success": True, "type": "legislation", "legislation_title": "Some Other Act 2001", "section": "1", "resolved_url": "https://www.legislation.gov.uk/search?title=X"}
        enrich_resolve_result(unknown, [SEARCH_ROW])
        self.assertEqual(unknown["resolved_url"], "https://www.legislation.gov.uk/search?title=X")
        case = {"success": True, "type": "case", "resolved_url": "https://caselaw.nationalarchives.gov.uk/uksc/2024/12"}
        enrich_resolve_result(case, [SEARCH_ROW])
        self.assertEqual(case["resolved_url"], "https://caselaw.nationalarchives.gov.uk/uksc/2024/12")


if __name__ == "__main__":
    unittest.main()
