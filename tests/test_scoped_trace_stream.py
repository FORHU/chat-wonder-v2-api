"""Session-scoped, customer-safe trace stream (GET /trace-stream/{session_id}).

The unscoped /trace-stream is the developer tracer's feed: every session's events, raw `text`
included, no auth. The scoped stream is what ilovelawyer-api consumes on a customer's behalf, so
what matters here is what it refuses to carry: other sessions, metrics, developer-grade `text`,
and any caller without the shared key.
"""

import asyncio
import json
import os
import threading
import unittest
from unittest.mock import patch

from fastapi import HTTPException

import the_server as srv


def _raw(**overrides):
    event = {"type": "cognition", "text": "internal detail", "summary": "The AI reasoned.", "session_id": "s1", "turn_id": "t1", "ts": 1.0}
    event.update(overrides)
    return json.dumps(event)


class UserTraceEventTests(unittest.TestCase):
    def test_keeps_only_the_customer_safe_fields(self):
        out = json.loads(srv._user_trace_event(_raw(), "s1"))
        self.assertEqual(out, {"type": "cognition", "summary": "The AI reasoned.", "turn_id": "t1", "ts": 1.0})

    def test_never_carries_the_developer_text_or_session_id(self):
        out = srv._user_trace_event(_raw(text="session s1 raw tool output"), "s1")
        self.assertNotIn("raw tool output", out)
        self.assertNotIn("session_id", json.loads(out))

    def test_drops_other_sessions(self):
        self.assertIsNone(srv._user_trace_event(_raw(session_id="s2"), "s1"))

    def test_drops_events_with_no_session(self):
        self.assertIsNone(srv._user_trace_event(_raw(session_id=None), "s1"))

    def test_drops_metrics_and_unknown_types(self):
        self.assertIsNone(srv._user_trace_event(_raw(type="metric"), "s1"))
        self.assertIsNone(srv._user_trace_event(_raw(type="something-new"), "s1"))

    def test_drops_events_without_a_plain_language_summary(self):
        self.assertIsNone(srv._user_trace_event(_raw(summary=None), "s1"))
        self.assertIsNone(srv._user_trace_event(_raw(summary=""), "s1"))

    def test_drops_unparseable_payloads(self):
        self.assertIsNone(srv._user_trace_event("not json", "s1"))


class RequireTraceKeyTests(unittest.TestCase):
    def test_fails_closed_when_no_key_is_configured(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("TRACE_STREAM_API_KEY", None)
            with self.assertRaises(HTTPException) as ctx:
                srv._require_trace_key("anything")
        self.assertEqual(ctx.exception.status_code, 503)

    def test_rejects_a_missing_or_wrong_key(self):
        with patch.dict(os.environ, {"TRACE_STREAM_API_KEY": "secret"}):
            for provided in (None, "", "wrong"):
                with self.assertRaises(HTTPException) as ctx:
                    srv._require_trace_key(provided)
                self.assertEqual(ctx.exception.status_code, 401)

    def test_accepts_the_right_key(self):
        with patch.dict(os.environ, {"TRACE_STREAM_API_KEY": "secret"}):
            srv._require_trace_key("secret")


class TurnIdStampingTests(unittest.TestCase):
    def setUp(self):
        self.state = srv.ChatState()
        self.state.trace_turn_id = "turn-42"
        srv._context.sessions["sess-stamp"] = self.state
        self.q = asyncio.Queue(maxsize=50)
        srv._trace_queues.add(self.q)

    def tearDown(self):
        srv._trace_queues.discard(self.q)
        srv._context.sessions.pop("sess-stamp", None)

    def _drain(self):
        out = []
        while not self.q.empty():
            out.append(json.loads(self.q.get_nowait()))
        return out

    def test_events_carry_the_sessions_current_turn_id(self):
        srv.broadcast_trace("action", "Executing `x`", "sess-stamp", summary="Running x.")
        self.assertEqual(self._drain()[0]["turn_id"], "turn-42")

    def test_a_worker_thread_event_is_stamped_too(self):
        # broadcast_trace is called from threads (the sync /chat path, tool workers), where a
        # ContextVar would not follow. The id comes off the session instead.
        t = threading.Thread(target=srv.broadcast_trace, args=("action", "from a thread", "sess-stamp"), kwargs={"summary": "Threaded."})
        t.start()
        t.join()
        self.assertEqual(self._drain()[0]["turn_id"], "turn-42")

    def test_the_next_turn_does_not_inherit_the_previous_id(self):
        self.state.trace_turn_id = None
        srv.broadcast_trace("action", "x", "sess-stamp", summary="Running x.")
        self.assertIsNone(self._drain()[0]["turn_id"])

    def test_sessions_restored_from_an_old_pickle_have_no_turn_id_attribute(self):
        del self.state.__dict__["trace_turn_id"]
        srv.broadcast_trace("action", "x", "sess-stamp", summary="Running x.")
        self.assertIsNone(self._drain()[0]["turn_id"])

    def test_events_with_no_session_have_no_turn_id(self):
        srv.broadcast_trace("request", "Install embeddings", None, summary="Uploading.")
        self.assertIsNone(self._drain()[0]["turn_id"])


class ScopedStreamEndpointTests(unittest.TestCase):
    def test_streams_only_the_requested_sessions_safe_events(self):
        async def run():
            with patch.dict(os.environ, {"TRACE_STREAM_API_KEY": "secret"}):
                resp = await srv.scoped_trace_stream("mine", x_api_key="secret")
            body = resp.body_iterator
            self.assertIn("connected", await body.__anext__())

            srv.broadcast_trace("cognition", "other user's text", "theirs", summary="Not yours.")
            srv.broadcast_trace("metric", "counter x", "mine", summary="metric summary")
            srv.broadcast_trace("action", "raw dump", "mine", summary="Looked up a statute.")

            chunk = await asyncio.wait_for(body.__anext__(), timeout=2)
            await body.aclose()
            return chunk

        chunk = asyncio.run(run())
        payload = json.loads(chunk[len("data: "):])
        self.assertEqual(payload["summary"], "Looked up a statute.")
        self.assertNotIn("Not yours", chunk)
        self.assertNotIn("raw dump", chunk)

    def test_refuses_without_the_key_before_streaming_anything(self):
        async def run():
            with patch.dict(os.environ, {"TRACE_STREAM_API_KEY": "secret"}):
                await srv.scoped_trace_stream("mine", x_api_key="nope")

        with self.assertRaises(HTTPException) as ctx:
            asyncio.run(run())
        self.assertEqual(ctx.exception.status_code, 401)


if __name__ == "__main__":
    unittest.main()
