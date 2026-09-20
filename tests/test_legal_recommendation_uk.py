"""cw#81 — get_legal_recommendation_uk: the UK sibling of get_legal_recommendation.

Unlike the Philippine tool (which has GPT-4o write an answer from the titles it found), this returns a research pack of
candidate UK legislation and case law with no generated text about the law, because the UK persona may only state law
it has fetched. No live UK Legal MCP or OpenAI call: the two search tools are stubbed on the user_functions module.
"""

import inspect
import json
import unittest
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)
import resources.functions.user_functions as uf  # noqa: E402  (loaded above)

TOOL = "get_legal_recommendation_uk"


def _legislation(n=1):
    return {
        "success": True,
        "results": [
            {"title": f"Act {i}", "type": "ukpga", "year": 1990 + i, "number": i, "url": f"https://www.legislation.gov.uk/ukpga/{1990 + i}/{i}",
             "content": "BODY TEXT THAT MUST NOT BE RETURNED", "score": 0.9}
            for i in range(n)
        ],
    }


def _case_law(n=1):
    return {
        "success": True,
        "results": [
            {"title": f"Smith v Jones {i}", "court": "UKSC", "date": "2020-01-01", "citation": f"[2020] UKSC {i}",
             "url": f"https://caselaw.nationalarchives.gov.uk/uksc/2020/{i}", "summary": "SUMMARY THAT MUST NOT BE RETURNED"}
            for i in range(n)
        ],
    }


class _Searches:
    """Stubs uf.legislation_search / uf.case_law_search and records how they were called."""

    def __init__(self, legislation, case_law):
        self.calls = {"legislation": [], "case_law": []}
        self._patches = [
            patch.object(uf, "legislation_search", lambda **kw: (self.calls["legislation"].append(kw), legislation)[1]),
            patch.object(uf, "case_law_search", lambda **kw: (self.calls["case_law"].append(kw), case_law)[1]),
        ]

    def __enter__(self):
        for p in self._patches:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()


class GetLegalRecommendationUkTests(unittest.TestCase):
    def test_returns_legislation_and_case_law_with_identifying_fields_only(self):
        with _Searches(_legislation(2), _case_law(2)):
            r = uf.get_legal_recommendation_uk(legal_issue="unfair dismissal")
        self.assertTrue(r["success"])
        self.assertEqual(r["issue"], "unfair dismissal")
        self.assertEqual([x["title"] for x in r["legislation"]], ["Act 0", "Act 1"])
        self.assertEqual(r["case_law"][0]["citation"], "[2020] UKSC 0")
        self.assertIn("url", r["legislation"][0])
        blob = json.dumps(r)
        self.assertNotIn("MUST NOT BE RETURNED", blob)  # snippet/body text never leaves the tool

    def test_keeps_at_most_five_rows_per_source(self):
        with _Searches(_legislation(9), _case_law(9)):
            r = uf.get_legal_recommendation_uk(legal_issue="tenancy deposit")
        self.assertEqual((len(r["legislation"]), len(r["case_law"])), (5, 5))

    def test_searches_both_sources_with_the_issue_and_a_limit(self):
        with _Searches(_legislation(), _case_law()) as s:
            uf.get_legal_recommendation_uk(legal_issue="notice period", user_context="employee for 3 years")
        self.assertEqual(s.calls["legislation"], [{"query": "notice period", "limit": 5}])
        self.assertEqual(s.calls["case_law"], [{"query": "notice period", "limit": 5}])

    def test_never_calls_an_llm(self):
        with _Searches(_legislation(), _case_law()), patch("openai.OpenAI", side_effect=AssertionError("must not call OpenAI")):
            r = uf.get_legal_recommendation_uk(legal_issue="notice period")
        self.assertTrue(r["success"])

    def test_missing_issue_is_an_error_and_does_not_search(self):
        with _Searches(_legislation(), _case_law()) as s:
            r = uf.get_legal_recommendation_uk(legal_issue="  ")
        self.assertFalse(r["success"])
        self.assertEqual(s.calls, {"legislation": [], "case_law": []})

    def test_one_failing_search_still_returns_the_other_and_reports_the_failure(self):
        with _Searches({"success": False, "error": "timeout"}, _case_law()):
            r = uf.get_legal_recommendation_uk(legal_issue="negligence")
        self.assertTrue(r["success"])
        self.assertEqual(r["legislation"], [])
        self.assertEqual(len(r["case_law"]), 1)
        self.assertEqual(r["search_errors"], {"legislation": "timeout"})

    def test_both_failing_is_a_failure(self):
        with _Searches({"success": False, "error": "a"}, {"success": False, "error": "b"}):
            r = uf.get_legal_recommendation_uk(legal_issue="negligence")
        self.assertFalse(r["success"])
        self.assertEqual(r["search_errors"], {"legislation": "a", "case_law": "b"})

    def test_tells_the_model_to_fetch_before_stating_law_and_carries_a_disclaimer(self):
        with _Searches(_legislation(), _case_law()):
            r = uf.get_legal_recommendation_uk(legal_issue="negligence")
        self.assertIn("fetch it", r["how_to_use"])
        self.assertIn("citations_resolve", r["how_to_use"])
        self.assertIn("not legal advice", r["disclaimer"])


class UkWiringTests(unittest.TestCase):
    @staticmethod
    def _tool_names(tag):
        _persona, _cleaned, tools, _addendum = srv.process_persona(tag)
        return {t["function"]["name"] for t in (tools or [])}

    def test_uk_persona_has_it_and_the_philippine_persona_does_not(self):
        self.assertIn(TOOL, self._tool_names("[legal ai uk] what are my rights"))
        self.assertNotIn(TOOL, self._tool_names("[legal ai] what are my rights"))

    def test_the_philippine_tool_is_still_philippine_only(self):
        self.assertIn("get_legal_recommendation", self._tool_names("[legal ai] what are my rights"))
        self.assertNotIn("get_legal_recommendation", self._tool_names("[legal ai uk] what are my rights"))

    def test_manifest_entry_requires_the_issue(self):
        tool = next(t["function"] for t in srv._context.all_fun_manifest if t["function"]["name"] == TOOL)
        self.assertEqual(tool["parameters"]["required"], ["legal_issue"])
        self.assertIn("NOT an answer", tool["description"])

    def test_uk_prompt_says_when_to_use_it_and_that_it_is_not_an_answer(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[legal ai uk] what are my rights")
        self.assertIn(TOOL, addendum)
        self.assertIn("starting list, not an answer", addendum)

    def test_results_feed_the_citation_pool_like_the_other_uk_search_tools(self):
        source = inspect.getsource(srv.execute_function_call)
        self.assertIn(f'"{TOOL}"', source)  # part of _UK_LEGAL_RESULT_TOOLS


class TraceTextTests(unittest.TestCase):
    def test_summary_counts_the_sources(self):
        text = srv._summarize_tool_result(TOOL, {"issue": "notice period", "legislation": [1, 2], "case_law": [1]})
        self.assertIn("2 legislation", text)
        self.assertIn("1 case law", text)

    def test_describe_and_label_use_the_issue(self):
        self.assertIn("notice period", srv._describe_tool_args(TOOL, json.dumps({"legal_issue": "notice period"})))
        self.assertIn("notice period", srv._trace_label_for_call(TOOL, {"legal_issue": "notice period"}))


if __name__ == "__main__":
    unittest.main()
