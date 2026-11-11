# -*- coding: utf-8 -*-
"""decisions_research.py (R2 prototype) with a fake Decisions API and a fake UK tools server --
no network. Live comparisons are run separately (see the module docstring)."""
import json
import os
import unittest
from unittest.mock import patch

import decisions_research as dr
import legal_responses_chain as lrc


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeDecisionsClient:
    """Answers act questions with `act_probs` (by act index) and section questions with
    `section_probs` (by section number). Records every request."""

    def __init__(self, act_probs, section_probs, fail=False):
        self.act_probs, self.section_probs, self.fail = act_probs, section_probs, fail
        self.requests = []

    def with_options(self, **_):
        return self

    def post(self, path, body=None, cast_to=None):
        assert path == "/decisions"
        self.requests.append(body)
        if self.fail:
            raise RuntimeError("decisions unavailable")
        answers = []
        for q in body["questions"]:
            if q["name"].startswith("act"):
                p = self.act_probs.get(int(q["name"][3:]), 0.0)
                answers.append({"name": q["name"], "type": "choice", "probabilities": [
                    {"value": "central", "probability": p}, {"value": "related", "probability": 1 - p},
                    {"value": "irrelevant", "probability": 0.0}]})
            else:
                section = q["instructions"].split("section ")[1].split(":")[0]
                answers.append({"name": q["name"], "type": "predicate",
                                "probability": self.section_probs.get(section, 0.0)})
        return _FakeResp({"answers": answers})


def fake_tools(calls):
    def run_tool(name, args):
        calls.append((name, args))
        if name == "legislation_get_toc":
            return {"success": True, "total_items": 4, "items": [
                "part-X: Unfair dismissal",
                "section-94: The right.",
                "section-108: Qualifying period of employment.",
                "section-123: Compensatory award.",
            ]}
        if name == "legislation_get_section":
            return {"success": True, "title": "t", "section_number": args["section"],
                    "content": f"text of s{args['section']}", "url": f"https://x/{args['section']}"}
        return None
    return run_tool


SEARCH = {"success": True, "results": [
    {"title": "Employment Rights Act 1996", "type": "ukpga", "year": 1996, "number": 18},
    {"title": "Some Unrelated Order 2026", "type": "uksi", "year": 2026, "number": 1},
]}


class PrefetchSectionsTests(unittest.TestCase):
    def setUp(self):
        dr.clear_toc_cache()

    def test_off_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("LEGAL_DECISIONS_PREFETCH", None)
            self.assertFalse(dr.prefetch_enabled())

    def test_picks_central_act_and_most_useful_sections(self):
        client = FakeDecisionsClient({0: 0.9, 1: 0.1}, {"108": 0.95, "123": 0.8, "94": 0.2})
        calls = []
        out = dr.prefetch_sections("legislation_search", SEARCH, "qualifying period?", client, fake_tools(calls))
        fetched_sections = [f["args"]["section"] for f in out["fetched"]]
        self.assertEqual(fetched_sections, ["108", "123"])  # ordered by probability; s94 below threshold
        self.assertEqual(out["metrics"]["opened"], 1)  # the unrelated Order is never opened
        self.assertEqual(len(client.requests), 2)  # one request for the Acts, one for all sections
        self.assertNotIn(("legislation_get_toc", {"type": "uksi"}), [(n, {"type": a.get("type")}) for n, a in calls if n != "legislation_get_toc"])

    def test_does_not_modify_the_search_result(self):
        result = json.loads(json.dumps(SEARCH))
        dr.prefetch_sections("legislation_search", result, "q", FakeDecisionsClient({0: 0.9}, {"108": 0.9}), fake_tools([]))
        self.assertEqual(result, SEARCH)

    def test_fails_open(self):
        out = dr.prefetch_sections("legislation_search", SEARCH, "q", FakeDecisionsClient({}, {}, fail=True), fake_tools([]))
        self.assertIsNone(out)

    def test_ignores_other_tools_and_empty_searches(self):
        client = FakeDecisionsClient({0: 0.9}, {"108": 0.9})
        self.assertIsNone(dr.prefetch_sections("case_law_search", SEARCH, "q", client, fake_tools([])))
        self.assertIsNone(dr.prefetch_sections("legislation_search", {"results": []}, "q", client, fake_tools([])))
        self.assertEqual(client.requests, [])


class TocCacheTests(unittest.TestCase):
    ACT = {"title": "Employment Rights Act 1996", "type": "ukpga", "year": 1996, "number": 18}

    def setUp(self):
        dr.clear_toc_cache()

    def toc_calls(self, calls):
        return sum(1 for name, _ in calls if name == "legislation_get_toc")

    def test_second_lookup_makes_no_tools_call(self):
        calls = []
        first, cached1 = dr._toc_sections(self.ACT, fake_tools(calls))
        second, cached2 = dr._toc_sections(self.ACT, fake_tools(calls))
        self.assertEqual((cached1, cached2), (False, True))
        self.assertEqual(self.toc_calls(calls), 1)
        self.assertEqual([s["section"] for s in second], [s["section"] for s in first])

    def test_cache_hit_reported_in_metrics(self):
        client = FakeDecisionsClient({0: 0.9}, {"108": 0.9})
        dr.prefetch_sections("legislation_search", SEARCH, "q", client, fake_tools([]))
        out = dr.prefetch_sections("legislation_search", SEARCH, "q", client, fake_tools([]))
        self.assertEqual(out["metrics"]["toc_cached"], 1)

    def test_failed_lookup_is_not_cached(self):
        calls = []
        failing = lambda name, args: (calls.append((name, args)) or None)
        self.assertEqual(dr._toc_sections(self.ACT, failing), ([], False))
        dr._toc_sections(self.ACT, fake_tools(calls))
        self.assertEqual(self.toc_calls(calls), 2)  # the failure didn't stick

    def test_expired_entry_is_refetched(self):
        calls = []
        with patch.dict(os.environ, {"LEGAL_TOC_CACHE_TTL_S": "60"}):
            dr._toc_sections(self.ACT, fake_tools(calls))
            with patch.object(dr.time, "monotonic", return_value=dr.time.monotonic() + 120):
                _, cached = dr._toc_sections(self.ACT, fake_tools(calls))
        self.assertFalse(cached)
        self.assertEqual(self.toc_calls(calls), 2)

    def test_size_limit_drops_the_oldest_act(self):
        with patch.dict(os.environ, {"LEGAL_TOC_CACHE_MAX": "1"}):
            other = {**self.ACT, "number": 99, "title": "Other Act"}
            dr._toc_sections(self.ACT, fake_tools([]))
            dr._toc_sections(other, fake_tools([]))
            self.assertIsNone(dr._cached_toc(self.ACT))
            self.assertIsNotNone(dr._cached_toc(other))


class DeliveryTests(unittest.TestCase):
    def test_section_key_ignores_max_chars_and_types(self):
        self.assertEqual(dr.section_key({"type": "ukpga", "year": 1996, "number": 18, "section": "108", "max_chars": 6000}),
                         dr.section_key({"type": "UKPGA", "year": "1996", "number": "18", "section": "108"}))
        self.assertIsNone(dr.section_key({"type": "ukpga", "year": 1996}))

    def test_delivers_as_real_get_section_round_trips_once(self):
        fetched = [{"args": {"type": "ukpga", "year": 1996, "number": 18, "section": "108", "max_chars": 6000},
                    "result": {"success": True, "content": "x"}, "label": "ERA s108"}]
        items, chains, log, keys = [], [], [], set()
        delivered = lrc._deliver_prefetched(fetched, items, chains, log, keys, lambda n, r: "summary")
        self.assertEqual(len(delivered), 1)
        self.assertEqual([i["type"] for i in items], ["function_call", "function_call_output"])
        self.assertEqual(items[0]["name"], "legislation_get_section")
        self.assertEqual(items[0]["call_id"], items[1]["call_id"])
        self.assertTrue(log[0]["prefetched"])
        self.assertEqual(chains[0]["name"], "legislation_get_section")
        # Delivering the same section again (a later search) adds nothing.
        self.assertEqual(lrc._deliver_prefetched(fetched, items, chains, log, keys, lambda n, r: ""), [])
        self.assertEqual(len(items), 2)


if __name__ == "__main__":
    unittest.main()
