"""Tests for the "stop generating on user Stop" feature (Phases 1-3).

No live OpenAI calls: 4.1-4.3 exercise _astream_llm/_watch_for_stop directly against fakes;
4.4-4.5 drive the real /chat-stream websocket handler through FastAPI's TestClient, with
state.openai_client replaced by a fake client so the LLM side is hermetic. init_openai_client
is patched to a no-op for the same reason -- it otherwise unconditionally overwrites
state.openai_client with a real OpenAI(...) client on every turn.
"""

import asyncio
import json
import threading
import time
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import the_server as srv


# ---------------------------------------------------------------------------
# Fakes shared by the integration tests (4.4/4.5)
# ---------------------------------------------------------------------------

class _FakeDelta:
    def __init__(self, content=None):
        self.content = content
        self.tool_calls = None
        self.function_call = None


class _FakeChunk:
    def __init__(self, content=None):
        self.choices = [SimpleNamespace(delta=_FakeDelta(content))]


class _SlowFakeStream:
    """Iterable Chat Completions stream stand-in with a close() spy. Yields plain-text
    chunks (no newlines) with a delay between them, so a Stop has time to land mid-stream."""

    def __init__(self, parts=None, delay=0.05):
        self.parts = parts if parts is not None else [f"word{i} " for i in range(40)]
        self.delay = delay
        self.closed = threading.Event()

    def __iter__(self):
        for part in self.parts:
            if self.closed.is_set():
                return
            time.sleep(self.delay)
            yield _FakeChunk(part)

    def close(self):
        self.closed.set()


class _FakeOpenAIClient:
    """Stands in for state.openai_client. Each .chat.completions.create() call makes a
    fresh _SlowFakeStream and remembers it as .last_stream, so a test can inspect whether
    that particular stream's close() was called."""

    def __init__(self, delay=0.05):
        self.delay = delay
        self.last_stream = None
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.last_stream = _SlowFakeStream(delay=self.delay)
        return self.last_stream


# ---------------------------------------------------------------------------
# 4.1 / 4.2: _astream_llm against a fake slow stream
# ---------------------------------------------------------------------------

class AstreamLlmTests(unittest.IsolatedAsyncioTestCase):
    async def test_cancel_event_closes_stream_within_a_second_no_chunks_after(self):
        stream = _SlowFakeStream(delay=0.03)
        cancel_event = srv.Event()
        token = srv._turn_cancel_event.set(cancel_event)
        try:
            received = []
            t_cancel = None
            async for chunk in srv._astream_llm(lambda msgs: stream, []):
                received.append(chunk)
                if len(received) == 3:
                    cancel_event.set()
                    t_cancel = time.time()
            elapsed = time.time() - t_cancel
        finally:
            srv._turn_cancel_event.reset(token)

        self.assertTrue(stream.closed.is_set())
        self.assertLess(elapsed, 1.0)
        # Nothing else should have been appended after the 3 we already had when we cancelled.
        self.assertEqual(len(received), 3)
        # The worker thread only notices the cancel between its time.sleep(delay) iterations
        # (see Known limits in the architecture doc) -- give it a moment to actually exit
        # before this test method returns and IsolatedAsyncioTestCase tears its loop down.
        await asyncio.sleep(stream.delay * 3)

    async def test_consumer_task_cancelled_still_closes_stream(self):
        stream = _SlowFakeStream(delay=0.03)

        async def consume():
            async for _ in srv._astream_llm(lambda msgs: stream, []):
                pass

        task = asyncio.create_task(consume())
        await asyncio.sleep(0.1)
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task

        self.assertTrue(stream.closed.is_set())
        await asyncio.sleep(stream.delay * 3)


# ---------------------------------------------------------------------------
# 4.3: _watch_for_stop against a fake websocket
# ---------------------------------------------------------------------------

class _FakeWebSocket:
    def __init__(self, messages):
        self._messages = list(messages)

    async def receive(self):
        if not self._messages:
            # Mirrors a broken/exhausted socket -- _watch_for_stop's except Exception
            # branch treats this the same as a disconnect.
            raise RuntimeError("no more messages")
        return self._messages.pop(0)


class _FakeTurnTask:
    def __init__(self):
        self.cancel_calls = 0

    def cancel(self):
        self.cancel_calls += 1


def _text_message(payload: dict):
    return {"type": "websocket.receive", "text": json.dumps(payload)}


class WatchForStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_disconnect_sets_event_and_cancels_turn(self):
        ws = _FakeWebSocket([{"type": "websocket.disconnect"}])
        cancel_event = srv.Event()
        turn_task = _FakeTurnTask()
        reason_box = [None]

        await srv._watch_for_stop(ws, cancel_event, turn_task, reason_box)

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(reason_box[0], "disconnect")
        self.assertEqual(turn_task.cancel_calls, 1)

    async def test_stop_message_sets_event_and_cancels_turn(self):
        ws = _FakeWebSocket([_text_message({"type": "stop"})])
        cancel_event = srv.Event()
        turn_task = _FakeTurnTask()
        reason_box = [None]

        await srv._watch_for_stop(ws, cancel_event, turn_task, reason_box)

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(reason_box[0], "stop")
        self.assertEqual(turn_task.cancel_calls, 1)

    async def test_unrelated_message_mid_turn_is_ignored_then_stop_still_works(self):
        ws = _FakeWebSocket([
            _text_message({"type": "chat", "user_input": "should be ignored"}),
            _text_message({"type": "stop"}),
        ])
        cancel_event = srv.Event()
        turn_task = _FakeTurnTask()
        reason_box = [None]

        await srv._watch_for_stop(ws, cancel_event, turn_task, reason_box)

        self.assertTrue(cancel_event.is_set())
        self.assertEqual(reason_box[0], "stop")
        self.assertEqual(turn_task.cancel_calls, 1)


# ---------------------------------------------------------------------------
# 4.4 / 4.5: integration through the real /chat-stream websocket handler
# ---------------------------------------------------------------------------

def _new_session(persona_prefix=""):
    session_id = f"test-{threading.get_ident()}-{time.time()}"
    state = srv.ChatState()
    fake_client = _FakeOpenAIClient(delay=0.05)
    state.openai_client = fake_client
    srv._context.sessions[session_id] = state
    return session_id, state, fake_client


class ChatStreamIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(srv.app)
        # init_openai_client would otherwise clobber our fake client with a real one on
        # every turn; _legal_use_responses_api/make_legal_verifier are pinned to the plain
        # Chat Completions path with no verify round, so the legal turn below only ever
        # talks to our fake stream (see the module docstring).
        self._patches = [
            patch.object(srv, "init_openai_client", lambda state, api_key, base_url=None: None),
            patch.object(srv, "_legal_use_responses_api", lambda: False),
            patch.object(srv, "make_legal_verifier", lambda *a, **k: None),
        ]
        for p in self._patches:
            p.start()
            self.addCleanup(p.stop)

    def test_disconnect_mid_turn_skips_finalize_and_extras_and_closes_provider_stream(self):
        session_id, state, fake_client = _new_session()

        with patch.object(srv, "_finalize_legal_response") as finalize_mock, \
             patch.object(srv, "_generate_structured_data") as structured_mock, \
             patch.object(srv, "_generate_reasoning_explanation") as reasoning_mock, \
             patch.object(srv, "_generate_decision_records") as decisions_mock, \
             patch.object(srv, "_generate_audio_overview_script") as audio_mock:

            with self.client.websocket_connect("/chat-stream") as ws:
                ws.send_json({
                    "type": "chat",
                    "session_id": session_id,
                    # No [legal ai] tag content here -- the tag itself picks the persona;
                    # a plain, citation-free question keeps prepare_legal_turn a no-op.
                    "user_input": "[legal ai] Please just say hello and nothing else.",
                })
                # Let a few fake chunks stream so the turn is genuinely mid-flight. Generous:
                # the first turn in a process pays a one-off cost (e.g. langid's model load).
                deadline = time.time() + 5.0
                while time.time() < deadline and fake_client.last_stream is None:
                    time.sleep(0.05)
                self.assertIsNotNone(fake_client.last_stream, "LLM call never started")
                time.sleep(0.15)

                # Close explicitly (rather than just letting the `with` block exit) so we can
                # wait, still inside the block, for the disconnect to be handled and the
                # worker thread to actually finish -- exiting the block tears down this
                # session's own background event loop, and the worker thread only notices
                # close() between its time.sleep(delay) iterations (see Known limits in the
                # architecture doc), so it can still be mid-sleep when that loop goes away.
                ws.close()
                deadline = time.time() + 2.0
                while time.time() < deadline and not (fake_client.last_stream and fake_client.last_stream.closed.is_set()):
                    time.sleep(0.05)
                self.assertTrue(fake_client.last_stream.closed.is_set(), "provider stream was not closed on disconnect")
                time.sleep(fake_client.delay * 3)

            finalize_mock.assert_not_called()
            structured_mock.assert_not_called()
            reasoning_mock.assert_not_called()
            decisions_mock.assert_not_called()
            audio_mock.assert_not_called()

        # A cancelled turn must leave no trace in session history (3.1).
        self.assertEqual(state.prompt, [])
        self.assertEqual(state.generated, [])

    def test_stop_message_ends_turn_then_a_second_message_answers_normally(self):
        session_id, state, fake_client = _new_session()

        with self.client.websocket_connect("/chat-stream") as ws:
            ws.send_json({"type": "chat", "session_id": session_id, "user_input": "tell me a long story"})
            # Generous: the first turn in a process pays a one-off cost (e.g. langid's model load).
            deadline = time.time() + 5.0
            while time.time() < deadline and fake_client.last_stream is None:
                time.sleep(0.05)
            self.assertIsNotNone(fake_client.last_stream, "LLM call never started")
            first_stream = fake_client.last_stream
            time.sleep(0.15)  # a few chunks have streamed

            ws.send_json({"type": "stop"})

            # The first turn should end with __END__, socket still open (2.6/4.5).
            end_1 = ws.receive_text()
            self.assertEqual(end_1, srv._context.__END__)

            deadline = time.time() + 2.0
            while time.time() < deadline and not first_stream.closed.is_set():
                time.sleep(0.05)
            self.assertTrue(first_stream.closed.is_set())
            # Same grace period as the disconnect test above -- let the first turn's worker
            # thread actually exit before starting the second turn.
            time.sleep(fake_client.delay * 3)

            # No message was lost by cancelling the watcher mid-receive() -- the connection
            # is still usable for a normal second turn right after (2.6, proven here).
            ws.send_json({"type": "chat", "session_id": session_id, "user_input": "hello again"})

            texts = []
            end_2_seen = False
            deadline = time.time() + 5.0
            while time.time() < deadline and not end_2_seen:
                msg = ws.receive_text()
                if msg == srv._context.__END__:
                    end_2_seen = True
                    break
                texts.append(msg)

        self.assertTrue(end_2_seen, "second turn never reached __END__")
        self.assertTrue(texts, "second turn produced no content")
        # Only the second (completed) turn should be in history -- the stopped first turn
        # left nothing behind (3.1).
        self.assertEqual(len(state.prompt), 1)
        self.assertEqual(len(state.generated), 1)
        self.assertEqual(state.prompt[0], "hello again")


if __name__ == "__main__":
    unittest.main()
