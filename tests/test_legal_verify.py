"""Unit tests for the legal verify→refine loop.

Pure audit/feedback helpers are tested directly. The chain-level tests drive the
real tool-calling loops (Chat Completions + Responses API, sync + streaming)
against a fake OpenAI client whose first answer carries a phantom citation and
whose second answer is clean — asserting the model is shown the rejected draft
plus a [VERIFIER] message, and that the revised text is what ships.

No live OpenAI/juris.ph calls.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import the_server as srv
import legal_responses_chain
from legal_verify import (
    DRAFT_DISCARD,
    audit_legal_draft,
    build_verifier_feedback,
    collect_tool_result_citables,
    make_legal_verifier,
)

POOL = [
    {"id": "abc", "title": "Manalo v. Republic", "url": "https://juris.ph/case/abc", "snippet": "the twin requirements of due process are notice and hearing"},
    {"id": "ra1", "title": "RA 11232", "url": "https://juris.ph/republic-act/ra1"},
]

GOOD = "See [Manalo Jurisprudence](https://juris.ph/case/abc)."
BAD = "See [Manalo Jurisprudence](https://juris.ph/case/zzz-made-up)."
TRUNCATED = "See [Manalo Jurisprudence](https://juris.ph/case/...)."
BAD_QUOTE = "> the Court held that mandamus lies against a purely ministerial refusal to act on the application\n\nParaphrase."


class AuditTests(unittest.TestCase):
    def test_clean_draft_has_no_issues(self):
        r = audit_legal_draft(GOOD, POOL)
        self.assertFalse(r.has_issues)

    def test_phantom_href_flagged(self):
        r = audit_legal_draft(BAD, POOL)
        self.assertEqual(r.unverified_urls, [("Manalo Jurisprudence", "https://juris.ph/case/zzz-made-up")])
        self.assertEqual(r.truncated_urls, [])

    def test_truncated_href_flagged_separately(self):
        r = audit_legal_draft(TRUNCATED, POOL)
        self.assertEqual(len(r.truncated_urls), 1)
        self.assertEqual(r.unverified_urls, [])

    def test_unverified_blockquote_flagged(self):
        r = audit_legal_draft(BAD_QUOTE, POOL)
        self.assertEqual(len(r.unverified_quotes), 1)

    def test_grounded_blockquote_not_flagged(self):
        r = audit_legal_draft("> the twin requirements of due process are notice and hearing", POOL)
        self.assertEqual(r.unverified_quotes, [])

    def test_short_or_linked_blockquotes_exempt(self):
        r = audit_legal_draft("> short quote\n> see [Manalo](https://juris.ph/case/abc) for the rule", POOL)
        self.assertEqual(r.unverified_quotes, [])

    def test_audit_matches_what_finalizer_would_gate(self):
        """The audit must flag exactly the links the post-hoc gate would demote."""
        from legal_citations import gate_unverified_legal_urls

        text = GOOD + " " + BAD + " " + TRUNCATED
        r = audit_legal_draft(text, POOL)
        gated = gate_unverified_legal_urls(text, POOL)
        self.assertEqual(r.url_issue_count, 2)
        self.assertIn("(https://juris.ph/case/abc)", gated)
        self.assertNotIn("zzz-made-up", gated.replace("Manalo Jurisprudence", ""))

    def test_citables_pair_titles_with_urls(self):
        self.assertEqual(
            collect_tool_result_citables(POOL),
            [("Manalo v. Republic", "https://juris.ph/case/abc"), ("RA 11232", "https://juris.ph/republic-act/ra1")],
        )


class FeedbackTests(unittest.TestCase):
    def test_feedback_lists_failures_and_allowed_pool(self):
        r = audit_legal_draft(BAD + "\n" + BAD_QUOTE, POOL)
        msg = build_verifier_feedback(r)
        self.assertTrue(msg.startswith("[VERIFIER"))
        self.assertIn("zzz-made-up", msg)
        self.assertIn("mandamus lies", msg)
        self.assertIn("Manalo v. Republic — https://juris.ph/case/abc", msg)
        self.assertIn("Do not mention this check", msg)

    def test_allowed_pool_capped(self):
        pool = [{"title": f"T{i}", "url": f"https://juris.ph/case/{i}"} for i in range(40)]
        r = audit_legal_draft(BAD, pool)
        msg = build_verifier_feedback(r, max_allowed=5)
        self.assertIn("and 35 more", msg)


class MakeVerifierTests(unittest.TestCase):
    def _state(self):
        return SimpleNamespace(last_search_legal_results=list(POOL))

    def test_clean_draft_returns_none(self):
        verify = make_legal_verifier(self._state(), max_rounds=1)
        self.assertIsNone(verify(GOOD, 0))

    def test_bad_draft_returns_feedback_then_exhausts(self):
        verify = make_legal_verifier(self._state(), max_rounds=1)
        fb = verify(BAD, 0)
        self.assertIsNotNone(fb)
        self.assertIn("[VERIFIER", fb.message)
        self.assertIn("1 unverified citation link(s)", fb.summary)
        # Second round with the same problem: rounds exhausted, finalizer takes over.
        self.assertIsNone(verify(BAD, 1))

    def test_zero_rounds_disables(self):
        self.assertIsNone(make_legal_verifier(self._state(), max_rounds=0))

    def test_env_rollback_lever(self):
        with patch.dict("os.environ", {"LEGAL_VERIFY_MAX_ROUNDS": "0"}):
            self.assertIsNone(make_legal_verifier(self._state()))
        with patch.dict("os.environ", {"LEGAL_VERIFY_MAX_ROUNDS": "2"}):
            verify = make_legal_verifier(self._state())
            self.assertIsNotNone(verify(BAD, 1))
            self.assertIsNone(verify(BAD, 2))

    def test_pool_read_at_call_time(self):
        """A revise round may retrieve the missing authority; the re-audit must see it."""
        state = self._state()
        verify = make_legal_verifier(state, max_rounds=1)
        self.assertIsNotNone(verify(BAD, 0))
        state.last_search_legal_results.append({"title": "Fetched", "url": "https://juris.ph/case/zzz-made-up"})
        self.assertIsNone(verify(BAD, 1))


# --- chain-level tests -------------------------------------------------------

def _cc_text_stream(text):
    delta = SimpleNamespace(content=text, tool_calls=None, function_call=None)
    return [SimpleNamespace(choices=[SimpleNamespace(delta=delta)])]


def _resp_text_stream(text):
    return [SimpleNamespace(type="response.output_text.delta", delta=text, output_index=0)]


class _FakeClient:
    """Scripted answers; records every request so tests can inspect the injected messages."""

    def __init__(self, answers, stream_factory):
        self.answers = list(answers)
        self.calls = []
        self._factory = stream_factory
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._factory(self.answers.pop(0))


def _state(client):
    st = srv.ChatState()
    st.openai_client = client
    st.last_search_legal_results = list(POOL)
    return st


def _collect(agen):
    async def run():
        out = []
        async for chunk in agen:
            out.append(chunk)
        return out

    return asyncio.run(run())


class ChainRefineTests(unittest.TestCase):
    MSGS = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]

    def _assert_second_call_saw_feedback(self, items):
        roles = [(m.get("role"), m.get("content", "")) for m in items if isinstance(m, dict)]
        self.assertEqual(roles[-2], ("assistant", BAD))
        self.assertEqual(roles[-1][0], "system")
        self.assertTrue(roles[-1][1].startswith("[VERIFIER"))

    def test_chat_completions_sync_revises_once(self):
        client = _FakeClient([BAD, GOOD], _cc_text_stream)
        state = _state(client)
        out = srv.run_function_chain(state, list(self.MSGS), max_chains=5, tools=[], auto_approval=True, verify=make_legal_verifier(state, max_rounds=1))
        self.assertEqual(out, GOOD)
        self.assertEqual(len(client.calls), 2)
        self._assert_second_call_saw_feedback(client.calls[1]["messages"])

    def test_responses_sync_revises_once(self):
        client = _FakeClient([BAD, GOOD], _resp_text_stream)
        state = _state(client)
        out = legal_responses_chain.run_function_chain_responses(state, list(self.MSGS), max_chains=5, tools=[], auto_approval=True, verify=make_legal_verifier(state, max_rounds=1))
        self.assertEqual(out, GOOD)
        self.assertEqual(len(client.calls), 2)
        self._assert_second_call_saw_feedback(client.calls[1]["input"])

    def test_responses_streaming_discards_draft_and_ships_revision(self):
        client = _FakeClient([BAD, GOOD], _resp_text_stream)
        state = _state(client)
        chunks = _collect(legal_responses_chain.streaming_run_function_chain_responses(state, list(self.MSGS), max_chains=5, tools=[], auto_approval=True, verify=make_legal_verifier(state, max_rounds=1)))
        self.assertIn(DRAFT_DISCARD, chunks)
        # What chat_stream would buffer: text after the discard, minus control/trace frames.
        after = chunks[chunks.index(DRAFT_DISCARD) + 1:]
        text = "".join(c for c in after if not c.startswith("[TRACE]"))
        self.assertEqual(text.strip(), GOOD)
        traces = [c for c in chunks if c.startswith("[TRACE]")]
        self.assertTrue(any('"phase": "start"' in t and "self_check" in t for t in traces))
        self.assertTrue(any('"phase": "result"' in t and "self_check" in t for t in traces))

    def test_chat_completions_streaming_discards_draft_and_ships_revision(self):
        client = _FakeClient([BAD, GOOD], _cc_text_stream)
        state = _state(client)
        chunks = _collect(srv.streaming_run_function_chain(state, list(self.MSGS), max_chains=5, tools=[], auto_approval=True, verify=make_legal_verifier(state, max_rounds=1)))
        self.assertIn(DRAFT_DISCARD, chunks)
        after = chunks[chunks.index(DRAFT_DISCARD) + 1:]
        text = "".join(c for c in after if not c.startswith("[TRACE]"))
        self.assertEqual(text.strip(), GOOD)

    def test_streaming_reyields_draft_when_revision_is_empty(self):
        """If the revise round produces nothing, the rejected draft must still ship
        (chat_stream dropped its buffer on DRAFT_DISCARD) so the finalizer gates it."""
        client = _FakeClient([BAD, ""], _resp_text_stream)
        state = _state(client)
        chunks = _collect(legal_responses_chain.streaming_run_function_chain_responses(state, list(self.MSGS), max_chains=5, tools=[], auto_approval=True, verify=make_legal_verifier(state, max_rounds=1)))
        after = chunks[chunks.index(DRAFT_DISCARD) + 1:]
        text = "".join(c for c in after if not c.startswith("[TRACE]"))
        self.assertEqual(text.strip(), BAD)

    def test_no_verifier_means_no_second_call(self):
        client = _FakeClient([BAD], _cc_text_stream)
        state = _state(client)
        out = srv.run_function_chain(state, list(self.MSGS), max_chains=5, tools=[], auto_approval=True)
        self.assertEqual(out, BAD)
        self.assertEqual(len(client.calls), 1)

    def test_draft_on_last_research_iteration_still_gets_revised(self):
        """The reserved iteration: max_chains=1 means one research iteration, but a
        verifier-driven revise round may still use the extra slot."""
        client = _FakeClient([BAD, GOOD], _cc_text_stream)
        state = _state(client)
        out = srv.run_function_chain(state, list(self.MSGS), max_chains=1, tools=[], auto_approval=True, verify=make_legal_verifier(state, max_rounds=1))
        self.assertEqual(out, GOOD)
        self.assertEqual(len(client.calls), 2)

    def test_reserved_iteration_not_usable_for_more_research(self):
        """Without a revise round in flight, the loop must stop at max_chains (tool cap unchanged)."""
        client = _FakeClient([GOOD, "unused"], _cc_text_stream)
        state = _state(client)
        out = srv.run_function_chain(state, list(self.MSGS), max_chains=1, tools=[], auto_approval=True, verify=make_legal_verifier(state, max_rounds=1))
        self.assertEqual(out, GOOD)
        self.assertEqual(len(client.calls), 1)


class PhPoolAccumulationTests(unittest.TestCase):
    """execute_function_call must never wipe earlier search rows or vetted get_* entries
    when the model runs another search — the live failure mode behind 17-20 demoted
    citations per PH turn."""

    def _run(self, session_id, name, result):
        with patch.dict(srv.__dict__, {name: lambda **kw: result}):
            return srv.execute_function_call({"name": name, "arguments": "{}"}, session_id=session_id)

    def test_second_search_keeps_first_search_rows_and_vetted_entries(self):
        sid = "pool-accumulate"
        state = srv.ChatState()
        srv._context.sessions[sid] = state
        try:
            self._run(sid, "search_jurisprudence", {"success": True, "results": [{"id": "a", "url": "https://juris.ph/case/a"}]})
            self._run(sid, "get_case", {"success": True, "id": "a", "title": "A", "url": "https://juris.ph/case/a", "type": "jurisprudence"})
            self._run(sid, "search_jurisprudence", {"success": True, "results": [{"id": "b", "url": "https://juris.ph/case/b"}, {"id": "a", "url": "https://juris.ph/case/a"}]})
            urls = [r.get("url") for r in state.last_search_legal_results]
            self.assertEqual(urls, ["https://juris.ph/case/a", "https://juris.ph/case/a", "https://juris.ph/case/b"])
            # Vetted get_case entry survives and the duplicate search row for "a" was not re-added.
            self.assertTrue(any(r.get("title") == "A" for r in state.last_search_legal_results))
        finally:
            srv._context.sessions.pop(sid, None)


if __name__ == "__main__":
    unittest.main()
