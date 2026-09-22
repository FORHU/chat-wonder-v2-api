"""legislation_search wired to plan_uk_legislation_query (FORHU/chat-wonder-v2-api#94) —
_uk_call is mocked, so this checks the query sent and result reordering, not uk-legal-mcp itself."""

import unittest
from unittest.mock import patch

import resources.functions.user_functions as uf


class LegislationSearchTests(unittest.TestCase):
    def _mock_uk_call(self, handler):
        def fake(tool_name, args):
            self.assertEqual(tool_name, "legislation_search")
            return {"results": handler(args["query"])}

        return fake

    def test_ordinary_query_passes_through_unchanged(self):
        sent = []

        def handler(q):
            sent.append(q)
            return [{"title": "Consumer Rights Act 2015", "type": "ukpga", "year": 2015, "number": 15}]

        with patch.object(uf, "_uk_call", side_effect=self._mock_uk_call(handler)):
            res = uf.legislation_search(query="consumer protection for faulty goods", limit=5)

        self.assertEqual(sent, ["consumer protection for faulty goods"])
        self.assertTrue(res["success"])

    def test_hra_is_expanded_before_the_call_and_never_sent_as_the_acronym(self):
        sent = []

        def handler(q):
            sent.append(q)
            return [{"title": "Human Rights Act 1998", "type": "ukpga", "year": 1998, "number": 42}]

        with patch.object(uf, "_uk_call", side_effect=self._mock_uk_call(handler)):
            res = uf.legislation_search(query="HRA", limit=5)

        self.assertEqual(sent, ["Human Rights Act 1998"])
        self.assertEqual(res["results"][0]["title"], "Human Rights Act 1998")

    def test_an_amending_act_no_longer_outranks_the_act_the_alias_meant(self):
        def handler(q):
            return [
                {"title": "Worker Protection (Amendment of Equality Act 2010) Act 2023", "type": "ukpga", "year": 2023, "number": 51},
                {"title": "Equality Act 2010", "type": "ukpga", "year": 2010, "number": 15},
            ]

        with patch.object(uf, "_uk_call", side_effect=self._mock_uk_call(handler)):
            res = uf.legislation_search(query="EA", limit=5)

        self.assertEqual(res["results"][0]["title"], "Equality Act 2010")
        self.assertEqual(len(res["results"]), 2)

    def test_acronym_inside_a_longer_query_is_left_alone(self):
        sent = []

        def handler(q):
            sent.append(q)
            return []

        with patch.object(uf, "_uk_call", side_effect=self._mock_uk_call(handler)):
            uf.legislation_search(query="HRA damages claim", limit=5)

        self.assertEqual(sent, ["HRA damages claim"])

    def test_a_transport_failure_surfaces_as_a_normal_tool_error(self):
        with patch.object(uf, "_uk_call", side_effect=RuntimeError("uk-legal-mcp unreachable")):
            res = uf.legislation_search(query="HRA", limit=5)
        self.assertFalse(res["success"])
        self.assertIn("error", res)

    def test_missing_query_is_still_rejected_before_any_call(self):
        with patch.object(uf, "_uk_call") as mocked:
            res = uf.legislation_search(query="   ", limit=5)
        self.assertFalse(res["success"])
        mocked.assert_not_called()


if __name__ == "__main__":
    unittest.main()
