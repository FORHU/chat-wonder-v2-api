"""#104 acceptance: /chat returns `usage`, and the /chat-stream websocket sends a `usage`
frame before [DONE]. The real handlers, tracker and client proxy run; only the model is faked.
No live OpenAI call."""

import asyncio
import json
import unittest
from types import SimpleNamespace as NS
from unittest import mock

from starlette.websockets import WebSocketDisconnect, WebSocketState

import llm_usage
import the_server as srv


def _inner_client():
    """Fake OpenAI client: every chat call reports 100 prompt (40 cached) / 10 completion tokens."""
    def create(**kw):
        return NS(model="gpt-5.6-luna",
                  usage=NS(prompt_tokens=100, completion_tokens=10, total_tokens=110,
                           prompt_tokens_details=NS(cached_tokens=40)),
                  choices=[NS(message=NS(content="ok"), finish_reason="stop")])
    return NS(chat=NS(completions=NS(create=create)))


def _factory(**kw):
    return llm_usage.UsageTrackingOpenAI(_inner_client())


def _model_call(state):
    state.openai_client.chat.completions.create(model="gpt-5.6-luna", messages=[])


class _Env(unittest.TestCase):
    def setUp(self):
        self.state = srv.ChatState()
        self.patches = [
            mock.patch.object(srv, "make_openai_client", _factory),
            mock.patch.object(srv._context, "openai_api_key", "k"),
            mock.patch.dict(srv._context.sessions, {"s1": self.state}),
            mock.patch.object(srv, "prepare_legal_turn", lambda text: (text, [])),
            mock.patch.object(srv, "_finalize_legal_response", lambda text, *a, **k: text),
            mock.patch.object(srv, "_generate_reasoning_explanation", lambda *a, **k: None),
            mock.patch.object(srv, "_generate_decision_records", lambda *a, **k: None),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)


class ChatRestUsageTests(_Env):
    def test_legal_chat_response_includes_usage(self):
        def fake_reason_loop(state, user_input, **kw):
            llm_usage.current().add_pass(); _model_call(state)
            llm_usage.current().add_pass(); _model_call(state)
            return "the answer"

        with mock.patch.object(srv, "reason_loop", fake_reason_loop):
            out = srv.chat(srv.ChatRequest(session_id="s1", user_input="[legal ai] what is estoppel"))
        u = out["usage"]
        self.assertEqual(out["response"], "the answer")
        self.assertEqual((u["prompt_tokens"], u["cached_input_tokens"], u["completion_tokens"]), (200, 80, 20))
        self.assertEqual((u["call_count"], u["pass_count"], u["model"]), (2, 2, "gpt-5.6-luna"))

    def test_usage_resets_between_turns(self):
        with mock.patch.object(srv, "reason_loop", lambda state, *a, **k: (_model_call(state), "a")[1]):
            srv.chat(srv.ChatRequest(session_id="s1", user_input="[legal ai] one"))
            out = srv.chat(srv.ChatRequest(session_id="s1", user_input="[legal ai] two"))
        self.assertEqual(out["usage"]["call_count"], 1)


class _FakeWS:
    client_state = WebSocketState.CONNECTED

    def __init__(self, messages):
        self._messages = list(messages)
        self.sent = []

    async def accept(self):
        pass

    async def receive_text(self):
        if not self._messages:
            raise WebSocketDisconnect()
        return json.dumps(self._messages.pop(0))

    async def send_text(self, text):
        self.sent.append(text)


class ChatStreamUsageTests(_Env):
    def test_usage_frame_comes_before_done(self):
        async def fake_stream(state, user_input, **kw):
            llm_usage.current().add_pass()
            await asyncio.to_thread(_model_call, state)   # a call from a worker thread
            yield "the answer"

        def fake_structured(*a, **k):
            _model_call(self.state)                       # an extra, run via asyncio.to_thread
            return None

        ws = _FakeWS([{"type": "chat", "session_id": "s1", "user_input": "[legal ai] what is estoppel"}])
        with mock.patch.object(srv, "streaming_reason_loop", fake_stream), \
                mock.patch.object(srv, "_generate_structured_data", fake_structured):
            asyncio.run(srv.chat_stream(ws))

        self.assertIn("[DONE]", ws.sent)
        done_at = ws.sent.index("[DONE]")
        usage_idx = [i for i, t in enumerate(ws.sent) if t.startswith("{") and '"type": "usage"' in t]
        self.assertEqual(len(usage_idx), 1, ws.sent)
        self.assertLess(usage_idx[0], done_at)
        frame = json.loads(ws.sent[usage_idx[0]])
        self.assertEqual(frame["session_id"], "s1")
        self.assertEqual(frame["data"]["call_count"], 2)        # loop call + extra, both counted
        self.assertEqual(frame["data"]["cached_input_tokens"], 80)
        self.assertEqual(frame["data"]["pass_count"], 1)


class AstreamLlmThreadHopTests(unittest.TestCase):
    """_astream_llm runs the model call on a plain Thread, which does not inherit ContextVars.
    It must run under a copied context or that call's tokens never reach the turn's usage."""

    def test_call_on_the_worker_thread_lands_in_the_turn(self):
        def perform_chat(messages):
            llm_usage.current().add(NS(prompt_tokens=7, completion_tokens=3), "gpt-5.6-luna")
            yield "chunk"

        async def run():
            turn = llm_usage.start_turn()
            out = [c async for c in srv._astream_llm(perform_chat, [])]
            return turn, out

        turn, out = asyncio.run(run())
        self.assertEqual(out, ["chunk"])
        self.assertEqual(turn.to_dict()["total_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
