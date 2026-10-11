import contextvars
import threading
import unittest
from types import SimpleNamespace as NS

import llm_usage
from legal_responses_chain import create_response


def cc_usage(p, c, cached=0):
    return NS(prompt_tokens=p, completion_tokens=c, total_tokens=p + c,
              prompt_tokens_details=NS(cached_tokens=cached))


class FakeCompletions:
    def __init__(self, stream_chunks=None):
        self.kwargs = None
        self.stream_chunks = stream_chunks

    def create(self, **kw):
        self.kwargs = kw
        if kw.get("stream"):
            return iter(self.stream_chunks)
        return NS(model="gpt-4o-mini", usage=cc_usage(100, 20, 40), choices=[NS()])


class FakeClient:
    def __init__(self, completions, responses=None):
        self.chat = NS(completions=completions)
        self.responses = responses
        self.other = "kept"


class TurnUsageTests(unittest.TestCase):
    def test_normalizes_both_schemes_and_splits_by_model(self):
        t = llm_usage.TurnUsage()
        t.add(cc_usage(100, 20, 40), "a")
        t.add(NS(input_tokens=10, output_tokens=5, input_tokens_details=NS(cached_tokens=2)), "b")
        t.add(None, "a")
        d = t.to_dict()
        self.assertEqual(d["prompt_tokens"], 110)
        self.assertEqual(d["cached_input_tokens"], 42)
        self.assertEqual(d["completion_tokens"], 25)
        self.assertEqual(d["total_tokens"], 135)
        self.assertEqual(d["call_count"], 3)
        self.assertEqual(d["model"], "a")
        self.assertEqual({r["model"] for r in d["by_model"]}, {"a", "b"})

    def test_start_turn_is_fresh(self):
        llm_usage.start_turn().add(cc_usage(5, 5), "m")
        self.assertEqual(llm_usage.start_turn().to_dict()["call_count"], 0)


class ProxyTests(unittest.TestCase):
    def test_non_stream_records_and_delegates(self):
        turn = llm_usage.start_turn()
        client = llm_usage.UsageTrackingOpenAI(FakeClient(FakeCompletions()))
        client.chat.completions.create(model="gpt-4o-mini", messages=[])
        self.assertEqual(turn.to_dict()["prompt_tokens"], 100)
        self.assertEqual(client.other, "kept")

    def test_stream_swallows_usage_chunk_and_requests_it(self):
        turn = llm_usage.start_turn()
        chunks = [NS(model="m", choices=[NS()], usage=None),
                  NS(model="m", choices=[], usage=cc_usage(7, 3))]
        comps = FakeCompletions(chunks)
        client = llm_usage.UsageTrackingOpenAI(FakeClient(comps))
        out = list(client.chat.completions.create(model="m", stream=True))
        self.assertEqual(len(out), 1)
        self.assertEqual(comps.kwargs["stream_options"], {"include_usage": True})
        self.assertEqual(turn.to_dict()["total_tokens"], 10)

    def test_stream_consumed_in_another_thread_still_records(self):
        turn = llm_usage.start_turn()
        chunks = [NS(model="m", choices=[], usage=cc_usage(1, 1))]
        stream = llm_usage.UsageTrackingOpenAI(FakeClient(FakeCompletions(chunks))).chat.completions.create(
            model="m", stream=True)
        t = threading.Thread(target=lambda: list(stream))
        t.start(); t.join()
        self.assertEqual(turn.to_dict()["call_count"], 1)

    def test_thread_under_copied_context_shares_turn(self):
        turn = llm_usage.start_turn()
        ctx = contextvars.copy_context()
        t = threading.Thread(target=lambda: ctx.run(lambda: llm_usage.current().add(cc_usage(2, 2), "m")))
        t.start(); t.join()
        self.assertEqual(turn.to_dict()["call_count"], 1)


class CreateResponseTests(unittest.TestCase):
    def test_stream_records_completed_and_forces_store_false(self):
        turn = llm_usage.start_turn()
        seen = {}
        events = [NS(type="response.output_text.delta"),
                  NS(type="response.completed",
                     response=NS(model="gpt-5.6-luna",
                                 usage=NS(input_tokens=50, output_tokens=10,
                                          input_tokens_details=NS(cached_tokens=30))))]

        def create(**kw):
            seen.update(kw)
            return iter(events)

        client = NS(responses=NS(create=create))
        out = list(create_response(client, model="x", stream=True, store=True))
        self.assertEqual(len(out), 2)
        self.assertIs(seen["store"], False)
        d = turn.to_dict()
        self.assertEqual((d["prompt_tokens"], d["cached_input_tokens"], d["model"]), (50, 30, "gpt-5.6-luna"))

    def test_non_stream_records(self):
        turn = llm_usage.start_turn()
        resp = NS(model="m", usage=NS(input_tokens=4, output_tokens=1))
        create_response(NS(responses=NS(create=lambda **kw: resp)), model="m")
        self.assertEqual(turn.to_dict()["total_tokens"], 5)


if __name__ == "__main__":
    unittest.main()
