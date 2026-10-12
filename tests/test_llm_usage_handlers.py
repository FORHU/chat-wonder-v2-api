"""#104: the handlers return / send the turn's usage. Real tracker and proxy, fake OpenAI inside.
No live OpenAI call."""

import asyncio
import json
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import llm_usage
import the_server as srv


def _fake_inner(content):
    def create(**kw):
        return NS(model="gpt-4o-mini",
                  usage=NS(prompt_tokens=120, completion_tokens=8, total_tokens=128,
                           prompt_tokens_details=NS(cached_tokens=64)),
                  choices=[NS(message=NS(content=content), finish_reason="stop")])
    return NS(chat=NS(completions=NS(create=create)))


class CategorizeUsageTests(unittest.TestCase):
    def test_response_carries_usage(self):
        factory = lambda **kw: llm_usage.UsageTrackingOpenAI(_fake_inner('{"category": "Demand Letter"}'))
        req = srv.CategorizeDocumentRequest(text="Dear sir, pay up.", filename="a.pdf")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"):
            out = asyncio.run(srv.categorize_document(req))
        self.assertEqual(out["category"], "Demand Letter")
        u = out["usage"]
        self.assertEqual((u["prompt_tokens"], u["cached_input_tokens"], u["completion_tokens"], u["call_count"]),
                         (120, 64, 8, 1))
        self.assertEqual(u["model"], "gpt-4o-mini")

    def test_each_request_starts_a_fresh_turn(self):
        factory = lambda **kw: llm_usage.UsageTrackingOpenAI(_fake_inner('{"category": "X"}'))
        req = srv.CategorizeDocumentRequest(text="t")
        with mock.patch.object(srv, "make_openai_client", factory), \
                mock.patch.object(srv._context, "openai_api_key", "k"):
            asyncio.run(srv.categorize_document(req))
            out = asyncio.run(srv.categorize_document(req))
        self.assertEqual(out["usage"]["call_count"], 1)

    def test_missing_key_still_returns_zero_usage(self):
        with mock.patch.object(srv._context, "openai_api_key", None):
            out = asyncio.run(srv.categorize_document(srv.CategorizeDocumentRequest(text="t")))
        self.assertFalse(out["success"])
        self.assertEqual(out["usage"]["call_count"], 0)


class SendUsageTests(unittest.TestCase):
    def test_frame_shape(self):
        llm_usage.start_turn().add(NS(prompt_tokens=5, completion_tokens=1), "m")
        sent = []

        class WS:
            async def send_text(self, t):
                sent.append(t)

        asyncio.run(srv._send_usage(WS(), "sess1"))
        frame = json.loads(sent[0])
        self.assertEqual((frame["type"], frame["session_id"]), ("usage", "sess1"))
        self.assertEqual(frame["data"]["prompt_tokens"], 5)

    def test_dead_socket_does_not_raise(self):
        class Dead:
            async def send_text(self, t):
                raise RuntimeError("closed")

        asyncio.run(srv._send_usage(Dead(), "s"))


if __name__ == "__main__":
    unittest.main()
