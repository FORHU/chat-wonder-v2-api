"""model_router.py (CW-6/CW-7): the decision logic only -- not the actual racing/orchestration,
which belongs in the_server.py's loop and is gated on a latency benchmark that doesn't exist
yet. Uses the same fake Decisions API client convention as tests/test_decisions_research.py.
"""

import os
import unittest
from unittest.mock import patch

import model_router as router
import routing_overrides as ro


class _FakeResp:
    def __init__(self, payload):
        self._payload = payload

    def json(self):
        return self._payload


class FakeDecisionsClient:
    """Answers model{i} predicates with `probs` (by candidate index). Records every request."""

    def __init__(self, probs, fail=False):
        self.probs, self.fail = probs, fail
        self.requests = []

    def with_options(self, **_):
        return self

    def post(self, path, body=None, cast_to=None):
        assert path == "/decisions"
        self.requests.append(body)
        if self.fail:
            raise RuntimeError("decisions unavailable")
        answers = [
            {"name": q["name"], "type": "predicate", "probability": self.probs.get(int(q["name"][len("model"):]), 0.0)}
            for q in body["questions"]
        ]
        return _FakeResp({"answers": answers})


def _no_override(*_a, **_k):
    return {"persona": "legal_uk", "routingMode": "AUTO", "manualPinnedModel": None, "toolOverrides": None, "source": "env"}


class DecideTurnModelTests(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("routing_overrides.get_override", side_effect=_no_override)
        self.patcher.start()
        self.broadcasts = []
        self.broadcast_patcher = patch("model_router._broadcast", side_effect=lambda **kw: self.broadcasts.append(kw))
        self.broadcast_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.broadcast_patcher.stop()

    def test_no_candidates_keeps_the_default_without_calling_decisions(self):
        client = FakeDecisionsClient({})
        result = router.decide_turn_model(client, "legal_uk", "gpt-5.6-luna", candidate_models=[], query="q")
        self.assertEqual(result, {"model": "gpt-5.6-luna", "decision_source": "default", "confidence": None})
        self.assertEqual(client.requests, [])

    def test_high_confidence_candidate_wins_and_broadcasts(self):
        client = FakeDecisionsClient({0: 0.9})
        result = router.decide_turn_model(client, "legal_uk", "gpt-5.6-luna", candidate_models=["gpt-6-luna"], query="q", session_id="s1")
        self.assertEqual(result, {"model": "gpt-6-luna", "decision_source": "race_corrected", "confidence": 0.9})
        self.assertEqual(len(self.broadcasts), 1)
        self.assertEqual(self.broadcasts[0]["to_model"], "gpt-6-luna")

    def test_low_confidence_keeps_the_default_and_still_broadcasts(self):
        # decide_turn_model's own docstring promises a broadcast "either way ... even when nothing
        # changed" -- CW-10's whole point is visibility into every turn-start decision, not just
        # the ones that switched models. Silence here would look identical to the decision never
        # having run at all, on the live trace.
        client = FakeDecisionsClient({0: 0.3})
        result = router.decide_turn_model(client, "legal_uk", "gpt-5.6-luna", candidate_models=["gpt-6-luna"], query="q")
        self.assertEqual(result["model"], "gpt-5.6-luna")
        self.assertEqual(result["decision_source"], "default")
        self.assertEqual(len(self.broadcasts), 1)
        self.assertEqual(self.broadcasts[0]["decision_source"], "default")
        self.assertEqual(self.broadcasts[0]["from_model"], self.broadcasts[0]["to_model"])

    def test_a_decisions_api_failure_never_raises_and_keeps_the_default(self):
        client = FakeDecisionsClient({}, fail=True)
        result = router.decide_turn_model(client, "legal_uk", "gpt-5.6-luna", candidate_models=["gpt-6-luna"], query="q")
        self.assertEqual(result["model"], "gpt-5.6-luna")
        self.assertEqual(result["decision_source"], "default")

    def test_a_candidate_that_needs_responses_api_is_dropped_when_the_call_cannot_switch(self):
        # gpt-5.6-luna requires /v1/responses for tools+reasoning (model_registry.ModelSpec).
        client = FakeDecisionsClient({0: 0.95})
        result = router.decide_turn_model(
            client, "legal_uk", "gpt-4o", candidate_models=["gpt-5.6-luna"], query="q", require_responses_api_ok=True,
        )
        self.assertEqual(result["model"], "gpt-4o")
        self.assertEqual(result["decision_source"], "default")
        self.assertEqual(client.requests, [])

    def test_candidate_equal_to_default_is_never_considered(self):
        client = FakeDecisionsClient({0: 1.0})
        result = router.decide_turn_model(client, "legal_uk", "gpt-5.6-luna", candidate_models=["gpt-5.6-luna"], query="q")
        self.assertEqual(result["decision_source"], "default")
        self.assertEqual(client.requests, [])


class ManualOverrideTests(unittest.TestCase):
    def test_turn_manual_override_short_circuits_before_any_decisions_call(self):
        def manual(*_a, **_k):
            return {"persona": "legal_uk", "routingMode": "MANUAL", "manualPinnedModel": "deepseek-chat", "toolOverrides": None, "source": "cache"}

        client = FakeDecisionsClient({0: 1.0})
        broadcasts = []
        with patch("routing_overrides.get_override", side_effect=manual), patch("model_router._broadcast", side_effect=lambda **kw: broadcasts.append(kw)):
            result = router.decide_turn_model(client, "legal_uk", "gpt-5.6-luna", candidate_models=["gpt-6-luna"], query="q")
        self.assertEqual(result, {"model": "deepseek-chat", "decision_source": "manual", "confidence": None})
        self.assertEqual(client.requests, [])
        self.assertEqual(broadcasts[0]["decision_source"], "manual")

    def test_tool_override_wins_over_persona_manual_pin(self):
        def override(*_a, **_k):
            return {
                "persona": "legal_uk", "routingMode": "MANUAL", "manualPinnedModel": "deepseek-chat",
                "toolOverrides": {"legislation_search": "gpt-4o"}, "source": "cache",
            }

        client = FakeDecisionsClient({})
        with patch("routing_overrides.get_override", side_effect=override), patch("model_router._broadcast"):
            result = router.decide_tool_model(client, "legal_uk", "legislation_search", "gpt-5.6-luna", query="q")
        self.assertEqual(result, {"model": "gpt-4o", "decision_source": "tool_override", "confidence": None})
        self.assertEqual(client.requests, [])


class DecideToolModelTests(unittest.TestCase):
    def setUp(self):
        self.patcher = patch("routing_overrides.get_override", side_effect=_no_override)
        self.patcher.start()
        self.broadcast_patcher = patch("model_router._broadcast")
        self.broadcast_patcher.start()

    def tearDown(self):
        self.patcher.stop()
        self.broadcast_patcher.stop()

    def test_an_unflagged_tool_makes_zero_decisions_calls(self):
        with patch.dict(os.environ, {"MODEL_SENSITIVE_TOOLS": ""}):
            client = FakeDecisionsClient({0: 1.0})
            result = router.decide_tool_model(client, "legal_uk", "legislation_search", "gpt-5.6-luna", candidate_models=["gpt-6-luna"], query="q")
        self.assertEqual(result["decision_source"], "default")
        self.assertEqual(client.requests, [])

    def test_a_flagged_tool_can_switch_on_high_confidence(self):
        with patch.dict(os.environ, {"MODEL_SENSITIVE_TOOLS": "legislation_search,get_case"}):
            client = FakeDecisionsClient({0: 0.8})
            result = router.decide_tool_model(client, "legal_uk", "legislation_search", "gpt-5.6-luna", candidate_models=["gpt-6-luna"], query="q")
        self.assertEqual(result["model"], "gpt-6-luna")
        self.assertEqual(result["decision_source"], "race_corrected")

    def test_a_flagged_tool_not_matching_this_call_is_unaffected(self):
        with patch.dict(os.environ, {"MODEL_SENSITIVE_TOOLS": "get_case"}):
            client = FakeDecisionsClient({0: 1.0})
            result = router.decide_tool_model(client, "legal_uk", "legislation_search", "gpt-5.6-luna", candidate_models=["gpt-6-luna"], query="q")
        self.assertEqual(result["decision_source"], "default")
        self.assertEqual(client.requests, [])


if __name__ == "__main__":
    unittest.main()
