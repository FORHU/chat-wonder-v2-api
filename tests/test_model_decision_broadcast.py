"""broadcast_model_decision (CW-6/CW-7's SCL hook): a model-routing decision reaches the same
Glass-box channel scl-core-v2 already consumes via /trace-stream, instead of a new event bus.
No decide_turn_model/decide_tool_model exists yet -- this is the broadcast half, built ahead of
the decision logic per the CW-10 SCL note, ready for model_router.py to call once it lands.
"""

import asyncio
import json
import unittest

import the_server as srv


class BroadcastModelDecisionTests(unittest.TestCase):
    def setUp(self):
        self.q = asyncio.Queue(maxsize=50)
        srv._trace_queues.add(self.q)

    def tearDown(self):
        srv._trace_queues.discard(self.q)

    def _drain(self):
        out = []
        while not self.q.empty():
            out.append(json.loads(self.q.get_nowait()))
        return out

    def test_emits_a_model_event_with_the_routing_fields_in_text(self):
        srv.broadcast_model_decision(
            persona="legal_uk", decision_kind="turn", from_model="gpt-5.6-luna", to_model="gpt-6-luna",
            decision_source="race_corrected", session_id="s1", confidence=0.82,
        )
        event = self._drain()[0]
        self.assertEqual(event["type"], "model")
        self.assertIn("persona=legal_uk", event["text"])
        self.assertIn("from_model=gpt-5.6-luna", event["text"])
        self.assertIn("to_model=gpt-6-luna", event["text"])
        self.assertIn("decision_source=race_corrected", event["text"])
        self.assertIn("confidence=0.82", event["text"])

    def test_switched_model_summary_names_both_models(self):
        srv.broadcast_model_decision(
            persona="legal_uk", decision_kind="turn", from_model="gpt-5.6-luna", to_model="gpt-6-luna",
            decision_source="race_corrected", session_id="s1",
        )
        self.assertEqual(
            self._drain()[0]["summary"],
            "Switched this turn from gpt-5.6-luna to gpt-6-luna (race_corrected).",
        )

    def test_kept_model_summary_names_the_one_model(self):
        srv.broadcast_model_decision(
            persona="legal_uk", decision_kind="turn", from_model="gpt-6-luna", to_model="gpt-6-luna",
            decision_source="default", session_id="s1",
        )
        self.assertEqual(self._drain()[0]["summary"], "Kept gpt-6-luna for this turn (default).")

    def test_tool_decision_names_the_tool_in_the_summary(self):
        srv.broadcast_model_decision(
            persona="legal_uk", decision_kind="tool", from_model="gpt-6-luna", to_model="deepseek-chat",
            decision_source="tool_override", session_id="s1", tool_name="legislation_search",
        )
        event = self._drain()[0]
        self.assertIn("tool=legislation_search", event["text"])
        self.assertEqual(
            event["summary"],
            "Switched the legislation_search tool call from gpt-6-luna to deepseek-chat (tool_override).",
        )

    def test_reaches_the_customer_safe_scoped_stream(self):
        # "model" must be in _USER_TRACE_TYPES, or chat-wonder's own model-decision broadcast
        # would be invisible to ilovelawyer-api's scoped /trace-stream consumer (CW-10).
        srv.broadcast_model_decision(
            persona="legal_uk", decision_kind="turn", from_model="gpt-5.6-luna", to_model="gpt-6-luna",
            decision_source="race_corrected", session_id="s1",
        )
        raw = self._drain()
        payload = srv._user_trace_event(json.dumps(raw[0]), "s1")
        self.assertIsNotNone(payload)
        safe = json.loads(payload)
        self.assertEqual(safe["type"], "model")
        self.assertNotIn("from_model", safe)  # the developer-grade `text` field never carries over
        self.assertIn("gpt-6-luna", safe["summary"])

    def test_events_with_no_session_still_broadcast(self):
        # Same posture as every other broadcast_trace call site: a session-less call (e.g. a
        # background benchmark run) still broadcasts, just with no session/turn to scope it to.
        srv.broadcast_model_decision(
            persona="legal_uk", decision_kind="turn", from_model="a", to_model="b", decision_source="default",
        )
        self.assertIsNone(self._drain()[0]["session_id"])


if __name__ == "__main__":
    unittest.main()
