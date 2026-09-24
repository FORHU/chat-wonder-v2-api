"""#73 — when the model leaves draft_pleading_uk's `authorities` empty after researching, the
server derives them from the session's fetched sources instead of relying on the model."""

import json
import unittest
from unittest.mock import patch

import the_server as srv
from uk_legal_mcp.authorities import authorities_from_pool

srv._load_user_functions(overwrite_globals=True)

ACT_ROW = {"title": "Late Payment of Commercial Debts (Interest) Act 1998", "url": "https://www.legislation.gov.uk/ukpga/1998/20"}
SECTION_8 = {
    "title": "Circumstances where statutory interest may be ousted or varied.",
    "section_number": "8",
    "content": "8 1 Any contract terms are void...",
    "url": "https://www.legislation.gov.uk/ukpga/1998/20/section/8",
}


class ExtractorTests(unittest.TestCase):
    def test_fetched_section_becomes_an_authority_with_the_act_title(self):
        pool = [{"results": [ACT_ROW]}, dict(SECTION_8)]
        self.assertEqual(
            authorities_from_pool(pool),
            [{"citation": "Late Payment of Commercial Debts (Interest) Act 1998, s 8",
              "proposition": "Circumstances where statutory interest may be ousted or varied"}],
        )

    def test_section_without_a_known_act_falls_back_to_the_url_identity(self):
        (a,) = authorities_from_pool([dict(SECTION_8)])
        self.assertEqual(a["citation"], "legislation.gov.uk/ukpga/1998/20, s 8")

    def test_bare_search_results_are_not_treated_as_authorities(self):
        pool = [{"results": [{"title": "Smith v Jones", "court": "UKSC", "citation": "[2020] UKSC 1"}]}]
        self.assertEqual(authorities_from_pool(pool), [])

    def test_fetched_judgment_header_is_included(self):
        (a,) = authorities_from_pool([{"title": "Smith v Jones", "citation": "[2020] UKSC 1"}])
        self.assertEqual(a["citation"], "Smith v Jones [2020] UKSC 1")

    def test_duplicates_collapse_and_empty_pool_is_empty(self):
        self.assertEqual(len(authorities_from_pool([dict(SECTION_8), dict(SECTION_8)])), 1)
        self.assertEqual(authorities_from_pool(None), [])


class HookTests(unittest.TestCase):
    def _run(self, args: dict, pool: list):
        sid = "test-authorities-fallback"
        srv._context.sessions[sid] = srv.ChatState()
        srv._context.sessions[sid].last_search_legal_results = pool
        seen = {}

        def fake(**kw):
            seen.update(kw)
            return {"success": True, "content": "x", "format": "docx", "pleading_type": "Particulars of Claim"}

        try:
            with patch.dict(srv.__dict__, {"draft_pleading_uk": fake}):
                srv.execute_function_call({"name": "draft_pleading_uk", "arguments": json.dumps(args)}, session_id=sid)
        finally:
            srv._context.sessions.pop(sid, None)
        return seen

    BASE = {"pleading_type": "Particulars of Claim", "case_facts": "f", "grounds": "g"}

    def test_empty_authorities_are_filled_from_the_session_pool(self):
        seen = self._run({**self.BASE, "authorities": []}, [{"results": [ACT_ROW]}, dict(SECTION_8)])
        self.assertEqual(seen["authorities"][0]["citation"], "Late Payment of Commercial Debts (Interest) Act 1998, s 8")

    def test_model_supplied_authorities_are_never_overridden(self):
        mine = [{"citation": "X v Y [2021] EWCA Civ 1", "proposition": "p"}]
        seen = self._run({**self.BASE, "authorities": mine}, [dict(SECTION_8)])
        self.assertEqual(seen["authorities"], mine)

    def test_no_research_in_session_leaves_authorities_absent(self):
        seen = self._run(dict(self.BASE), [])
        self.assertNotIn("authorities", seen)


class PrefillTests(unittest.TestCase):
    """The proposed call the tracer shows must already carry the filled authorities."""

    def _prefill(self, name: str, args: dict, pool: list) -> dict:
        sid = "test-authorities-prefill"
        srv._context.sessions[sid] = srv.ChatState()
        srv._context.sessions[sid].last_search_legal_results = pool
        fc = {"name": name, "arguments": json.dumps(args)}
        try:
            srv._prefill_uk_pleading_authorities(fc, sid)
        finally:
            srv._context.sessions.pop(sid, None)
        return json.loads(fc["arguments"])

    def test_proposed_call_shows_filled_authorities(self):
        out = self._prefill("draft_pleading_uk", {"grounds": "g", "authorities": []}, [{"results": [ACT_ROW]}, dict(SECTION_8)])
        self.assertEqual(out["authorities"][0]["citation"], "Late Payment of Commercial Debts (Interest) Act 1998, s 8")

    def test_other_tools_and_supplied_authorities_are_untouched(self):
        self.assertNotIn("authorities", self._prefill("draft_pleading", {"grounds": "g"}, [dict(SECTION_8)]))
        mine = [{"citation": "X v Y [2021] EWCA Civ 1", "proposition": "p"}]
        self.assertEqual(self._prefill("draft_pleading_uk", {"authorities": mine}, [dict(SECTION_8)])["authorities"], mine)


class EveryTraceSiteIsCoveredTests(unittest.TestCase):
    """Legal personas run on the Responses-API chain in legal_responses_chain.py, not only the
    loops in the_server.py. Every place that traces a proposed tool call must fill authorities
    first, or the tracer shows the model's raw (empty) arguments."""

    def test_every_proposed_tool_call_trace_is_preceded_by_the_prefill(self):
        import pathlib
        root = pathlib.Path(__file__).resolve().parent.parent
        for name in ("the_server.py", "legal_responses_chain.py"):
            lines = (root / name).read_text(encoding="utf-8").splitlines()
            sites = [i for i, ln in enumerate(lines) if "Proposed tool call:" in ln]
            self.assertTrue(sites, f"no trace sites found in {name}")
            for i in sites:
                window = "\n".join(lines[max(0, i - 6):i])
                self.assertIn("_prefill_uk_pleading_authorities", window, f"{name}:{i + 1} traces a proposed call without the prefill")


if __name__ == "__main__":
    unittest.main()
