"""search_republic_acts wired to plan_ra_query (FORHU/chat-wonder-v2-api#75) — _juris_call is
mocked, so this checks which queries run and how results are combined, not juris.ph itself."""

import unittest
from unittest.mock import patch

import resources.functions.user_functions as uf


class SearchRepublicActsTests(unittest.TestCase):
    def setUp(self):
        uf._search_cache.clear()

    def _mock_juris_call(self, by_query):
        def fake(tool_name, args):
            self.assertEqual(tool_name, "search_republic_acts")
            rows = by_query(args["query"])
            if rows == "down":
                raise RuntimeError("juris.ph unreachable")
            return {"results": rows}

        return fake

    def test_ordinary_query_runs_once(self):
        calls = []

        def by_query(q):
            calls.append(q)
            return [{"id": "a", "ra_number": "1"}]

        with patch.object(uf, "_juris_call", side_effect=self._mock_juris_call(by_query)):
            res = uf.search_republic_acts(query="violence against women", limit=5)

        self.assertEqual(calls, ["violence against women"])
        self.assertTrue(res["success"])
        self.assertEqual(res["search_type"], "juris_mcp")

    def test_vawc_fans_out_and_puts_ra_9262_first(self):
        calls = []

        def by_query(q):
            calls.append(q)
            if q == "RA 9262":
                return [{"id": "noise-1", "ra_number": "8796", "score": 0.9}]
            if q == "Republic Act No. 9262":
                return [
                    {"id": "ra9262", "ra_number": "9262", "score": 0.4},
                    {"id": "noise-1", "ra_number": "8796", "score": 0.9},
                ]
            return [{"id": "amend", "ra_number": "10398", "score": 0.8}]

        with patch.object(uf, "_juris_call", side_effect=self._mock_juris_call(by_query)):
            res = uf.search_republic_acts(query="VAWC", limit=5)

        self.assertEqual(
            sorted(calls),
            sorted([
                "Republic Act No. 9262",
                "R.A. No. 9262",
                "RA 9262",
                "Anti-Violence Against Women and Their Children Act of 2004",
            ]),
        )
        self.assertNotIn("VAWC", calls)
        self.assertEqual([r["id"] for r in res["results"]], ["ra9262", "noise-1", "amend"])
        self.assertEqual(res["query"], "VAWC")
        self.assertEqual(res["total_results"], 3)

    def test_merged_result_trimmed_to_limit_exact_number_kept(self):
        def by_query(q):
            if q == "RA 9160":
                return [{"id": "ra9160", "ra_number": "9160"}]
            return [{"id": "x1", "ra_number": "1"}, {"id": "x2", "ra_number": "2"}]

        with patch.object(uf, "_juris_call", side_effect=self._mock_juris_call(by_query)):
            res = uf.search_republic_acts(query="AMLA", limit=2)

        self.assertEqual([r["id"] for r in res["results"]], ["ra9160", "x1"])

    def test_eo_pd_ao_mo_return_no_results_and_never_call_juris(self):
        with patch.object(uf, "_juris_call") as mocked:
            for q, label in [
                ("PD 442", "Presidential Decree No. 442"),
                ("EO 209", "Executive Order No. 209"),
                ("AO 25", "Administrative Order No. 25"),
                ("MO 32", "Memorandum Order No. 32"),
            ]:
                res = uf.search_republic_acts(query=q, limit=5)
                self.assertTrue(res["success"])
                self.assertEqual(res["results"], [])
                self.assertEqual(res["total_results"], 0)
                self.assertIn(label, res["note"])
            mocked.assert_not_called()

    def test_two_spellings_that_expand_the_same_way_share_a_cache_entry(self):
        calls = []

        def by_query(q):
            calls.append(q)
            return [{"id": "ra9262", "ra_number": "9262"}]

        with patch.object(uf, "_juris_call", side_effect=self._mock_juris_call(by_query)):
            uf.search_republic_acts(query="VAWC", limit=5)
            call_count_after_first = len(calls)
            res2 = uf.search_republic_acts(query="vawc", limit=5)  # same expansion, different case

        self.assertTrue(res2["cached"])
        self.assertEqual(len(calls), call_count_after_first)  # no new upstream calls

    def test_an_acronym_and_its_bare_ra_number_are_cached_separately(self):
        """Different expansions (VAWC pulls in the act title too) -> different cache entries;
        each still resolves to the same act. See test_vawc_fans_out_and_puts_ra_9262_first."""
        calls = []

        def by_query(q):
            calls.append(q)
            return [{"id": "ra9262", "ra_number": "9262"}]

        with patch.object(uf, "_juris_call", side_effect=self._mock_juris_call(by_query)):
            uf.search_republic_acts(query="VAWC", limit=5)
            res2 = uf.search_republic_acts(query="RA 9262", limit=5)

        self.assertFalse(res2["cached"])
        self.assertEqual([r["id"] for r in res2["results"]], ["ra9262"])

    def test_a_transport_failure_surfaces_as_a_normal_tool_error(self):
        def by_query(q):
            return "down"

        with patch.object(uf, "_juris_call", side_effect=self._mock_juris_call(by_query)):
            res = uf.search_republic_acts(query="violence against women", limit=5)

        self.assertFalse(res["success"])
        self.assertIn("error", res)


if __name__ == "__main__":
    unittest.main()
