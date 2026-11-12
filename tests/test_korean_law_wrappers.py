"""Success/failure-shape tests for the Korean Law MCP tool wrappers in user_functions.py.

_kr_call is mocked throughout, so these check argument-building and error handling, not the
Korean Law MCP itself (see test_korean_law_mcp_client.py for the transport layer)."""

import unittest
from unittest.mock import patch

import resources.functions.user_functions as uf


class KoreanLawWrapperTests(unittest.TestCase):
    def test_search_law_requires_query(self):
        res = uf.search_law()
        self.assertFalse(res["success"])

    def test_search_law_passes_query_and_display(self):
        captured = {}

        def fake(tool_name, args):
            captured["tool_name"] = tool_name
            captured["args"] = args
            return {"results": [{"lawId": "001234"}]}

        with patch.object(uf, "_kr_call", side_effect=fake):
            res = uf.search_law(query="관세법", display=10)

        self.assertTrue(res["success"])
        self.assertEqual(captured["tool_name"], "search_law")
        self.assertEqual(captured["args"], {"query": "관세법", "display": 10})

    def test_get_law_text_requires_mst_or_lawid(self):
        res = uf.get_law_text()
        self.assertFalse(res["success"])

    def test_get_law_text_passes_jo_and_efyd(self):
        captured = {}

        def fake(tool_name, args):
            captured["args"] = args
            return {"text": "..."}

        with patch.object(uf, "_kr_call", side_effect=fake):
            res = uf.get_law_text(mst="123", jo="제38조", efYd="20230101")

        self.assertTrue(res["success"])
        self.assertEqual(captured["args"], {"mst": "123", "jo": "제38조", "efYd": "20230101"})

    def test_get_annexes_requires_lawname(self):
        res = uf.get_annexes()
        self.assertFalse(res["success"])

    def test_ordinance_radar_requires_at_least_one_identifier(self):
        res = uf.ordinance_radar()
        self.assertFalse(res["success"])

    def test_legal_research_requires_query_unless_document_review(self):
        res = uf.legal_research(task="law_system")
        self.assertFalse(res["success"])

    def test_legal_research_document_review_requires_text(self):
        res = uf.legal_research(task="document_review")
        self.assertFalse(res["success"])

    def test_legal_research_document_review_with_text_succeeds(self):
        with patch.object(uf, "_kr_call", return_value={"analysis": []}):
            res = uf.legal_research(task="document_review", text="some contract text")
        self.assertTrue(res["success"])

    def test_legal_analysis_requires_mode(self):
        res = uf.legal_analysis()
        self.assertFalse(res["success"])

    def test_legal_analysis_passes_mode(self):
        captured = {}

        def fake(tool_name, args):
            captured["args"] = args
            return {"verified": []}

        with patch.object(uf, "_kr_call", side_effect=fake):
            res = uf.legal_analysis(mode="verify_citations", text="민법 제750조")

        self.assertTrue(res["success"])
        self.assertEqual(captured["args"]["mode"], "verify_citations")

    def test_search_decisions_requires_domain(self):
        res = uf.search_decisions()
        self.assertFalse(res["success"])

    def test_search_decisions_passes_domain_and_query(self):
        captured = {}

        def fake(tool_name, args):
            captured["args"] = args
            return {"results": []}

        with patch.object(uf, "_kr_call", side_effect=fake):
            res = uf.search_decisions(domain="precedent", query="negligence")

        self.assertTrue(res["success"])
        self.assertEqual(captured["args"], {"domain": "precedent", "query": "negligence"})

    def test_get_decision_text_requires_domain_and_id(self):
        res = uf.get_decision_text(domain="precedent")
        self.assertFalse(res["success"])
        res = uf.get_decision_text(id="abc")
        self.assertFalse(res["success"])

    def test_discover_tools_kr_requires_intent(self):
        res = uf.discover_tools_kr()
        self.assertFalse(res["success"])

    def test_execute_tool_kr_requires_tool_name(self):
        res = uf.execute_tool_kr(params={"x": 1})
        self.assertFalse(res["success"])

    def test_wrapper_reports_kr_call_exception_as_failure(self):
        with patch.object(uf, "_kr_call", side_effect=RuntimeError("boom")):
            res = uf.search_law(query="민법")
        self.assertFalse(res["success"])
        self.assertIn("boom", res["error"])


if __name__ == "__main__":
    unittest.main()
