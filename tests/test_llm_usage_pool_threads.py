"""#104: a call made on a ThreadPoolExecutor worker must land in the turn's usage. A pool thread does
not inherit the turn's ContextVar, so the stylist persona's parallel outfit + cosmetics search submits
each job under its own copy of the context. This pins the idiom the server uses."""
import contextvars
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace as NS

import llm_usage


def _model_call():
    llm_usage.current().add(NS(prompt_tokens=10, completion_tokens=2), "gpt-4o-mini")


class PoolThreadUsageTests(unittest.TestCase):
    def test_plain_submit_loses_the_turn(self):
        turn = llm_usage.start_turn()
        with ThreadPoolExecutor(max_workers=1) as pool:
            try:
                pool.submit(_model_call).result()
            except AttributeError:
                pass  # no turn on the worker thread: current() is None there
        self.assertEqual(turn.to_dict()["call_count"], 0)

    def test_submit_under_a_copied_context_lands_in_the_turn(self):
        turn = llm_usage.start_turn()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(contextvars.copy_context().run, _model_call) for _ in range(2)]
            for job in jobs:
                job.result()
        usage = turn.to_dict()
        self.assertEqual(usage["call_count"], 2)
        self.assertEqual(usage["prompt_tokens"], 20)


if __name__ == "__main__":
    unittest.main()
