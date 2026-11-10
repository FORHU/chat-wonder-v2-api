"""Unit + regression tests for confidence.py and its integration points.

The regression test in ChatWonderStreamingTests is the one that matters most: it drives
the_server.py's real streaming_run_function_chain with fake logprobs and asserts the exact
failure mode from the project's handoff notes cannot recur -- [CONF] frames must never
accumulate into full_response, the way an earlier attempt's [METRIC] frames did (that bug sent
the whole buffered legal answer as one unparseable blob and dropped the connection).
"""

import asyncio
import json
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import confidence
import model_registry
import the_server as srv


# ---------------------------------------------------------------------------
# confidence.token_entropy
# ---------------------------------------------------------------------------

class TokenEntropyTests(unittest.TestCase):
    def test_bounded_by_ln5(self):
        # Five equally likely candidates is the maximum-entropy case for a top-5 renormalized
        # distribution -- H = ln(5) exactly.
        equal_logprob = math.log(0.2)
        h = confidence.token_entropy([equal_logprob] * 5)
        self.assertAlmostEqual(h, confidence.LN5, places=6)

    def test_confident_token_has_low_entropy(self):
        # One candidate overwhelmingly likely, four negligible -> entropy near zero.
        h = confidence.token_entropy([-0.001, -12.0, -14.0, -15.0, -16.0])
        self.assertLess(h, 0.05)

    def test_empty_or_non_finite_returns_none(self):
        self.assertIsNone(confidence.token_entropy([]))
        self.assertIsNone(confidence.token_entropy([float("nan"), float("inf"), None]))
        self.assertIsNone(confidence.token_entropy(None))

    def test_renormalizes_unlike_the_reference_remote_path(self):
        # The Thermo Chat reference's remote path sums raw probabilities WITHOUT renormalizing
        # (a partial-sum surprisal, unbounded relative to ln5). This module must renormalize:
        # three low-probability candidates still produce an entropy <= ln(3) once renormalized.
        h = confidence.token_entropy([-5.0, -5.0, -5.0])
        self.assertLessEqual(h, math.log(3) + 1e-6)


# ---------------------------------------------------------------------------
# confidence.group_words / entropy_payload
# ---------------------------------------------------------------------------

class GroupWordsTests(unittest.TestCase):
    def test_splits_on_leading_whitespace(self):
        entries = [("Hello", 0.1), (",", 0.2), (" world", 0.9), ("!", 0.05)]
        words = confidence.group_words(entries)
        self.assertEqual([w["w"] for w in words], ["Hello,", "world!"])

    def test_word_score_is_max_not_mean(self):
        # Matches the Thermo Chat reference's own display rule (and the preprint): a word's
        # score is the MAX entropy among its tokens, never an average.
        entries = [(" liab", 0.1), ("ility", 0.9)]
        words = confidence.group_words(entries)
        self.assertEqual(len(words), 1)
        self.assertAlmostEqual(words[0]["h"], 0.9)
        self.assertEqual(words[0]["n"], 2)

    def test_tokens_with_no_measured_entropy_are_none(self):
        entries = [(" word", None)]
        words = confidence.group_words(entries)
        self.assertIsNone(words[0]["h"])
        self.assertIsNone(words[0]["hn"])

    def test_empty_entries_produce_no_words(self):
        self.assertEqual(confidence.group_words([]), [])

    def test_entropy_payload_shape(self):
        payload = confidence.entropy_payload([(" hi", 0.5)])
        self.assertEqual(payload["unit"], "nats")
        self.assertEqual(payload["metric_source"], "top5_renormalized")
        self.assertAlmostEqual(payload["ceiling"], round(confidence.LN5, 4))
        self.assertEqual(len(payload["words"]), 1)

    def test_entropy_payload_none_when_nothing_measured(self):
        self.assertIsNone(confidence.entropy_payload([]))
        self.assertIsNone(confidence.entropy_payload(None))


# ---------------------------------------------------------------------------
# confidence.to_live_frame / to_frame — exact frame shape the websocket guards match on
# ---------------------------------------------------------------------------

class FrameShapeTests(unittest.TestCase):
    def test_live_frame_round_trips(self):
        frame = confidence.to_live_frame(" liability", 0.84)
        self.assertTrue(frame.startswith("[CONF]"))
        self.assertTrue(frame.endswith("[/CONF]"))
        body = json.loads(frame[len("[CONF]"):-len("[/CONF]")])
        self.assertEqual(body["token"], " liability")
        self.assertAlmostEqual(body["h"], 0.84)

    def test_build_payload_round_trips_through_to_frame(self):
        spans = [confidence.Span(quote="x", tier="UNSOURCED", source="provenance", why="no overlap")]
        payload = confidence.build_payload("legal", "gpt-5.6-luna", spans, checked=True)
        frame = confidence.to_frame(payload)
        parsed = json.loads(frame)
        self.assertEqual(parsed["type"], "confidence")
        self.assertEqual(parsed["spans"][0]["tier"], "UNSOURCED")


class CoverageHonestyTests(unittest.TestCase):
    """coverage is the field a client-facing UI reads to decide whether to say "shading
    unavailable" — these three states must stay distinguishable."""

    def test_none_when_nothing_ran(self):
        payload = confidence.build_payload("intake", "gpt-5.6-luna", [], checked=False, tokens=None)
        self.assertEqual(payload["coverage"], "none")

    def test_partial_when_only_one_signal_ran(self):
        payload = confidence.build_payload("legal", "gpt-4o-mini", [], checked=True, tokens=None)
        self.assertEqual(payload["coverage"], "partial")

    def test_full_when_both_signals_ran(self):
        payload = confidence.build_payload(
            "legal", "gpt-4o-mini", [], checked=True, tokens={"unit": "nats", "words": []},
        )
        self.assertEqual(payload["coverage"], "full")

    def test_clean_answer_with_no_issues_is_not_none(self):
        # A checked answer with zero flagged spans must read as "classified, nothing to flag"
        # (partial/full), never as "none" (which means the check never ran) — an empty span
        # list alone cannot distinguish these, which is exactly why `checked` is a separate
        # argument.
        payload = confidence.build_payload("legal_uk", "gpt-5.6-luna", [], checked=True)
        self.assertNotEqual(payload["coverage"], "none")


# ---------------------------------------------------------------------------
# confidence.provenance_spans — the formula (replaces judge-based attribution/self-consistency)
# ---------------------------------------------------------------------------

class ProvenanceSpansTests(unittest.TestCase):
    SOURCES = [
        "Section 103A of the Employment Rights Act 1996 provides that a dismissal is "
        "automatically unfair if the reason, or principal reason, for the dismissal is that "
        "the employee made a protected disclosure.",
    ]

    def test_near_verbatim_sentence_is_quoted(self):
        text = (
            "Section 103A of the Employment Rights Act 1996 provides that a dismissal is "
            "automatically unfair if the reason, or principal reason, for the dismissal is "
            "that the employee made a protected disclosure."
        )
        spans = confidence.provenance_spans(text, self.SOURCES)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].tier, "QUOTED")

    def test_paraphrase_with_shared_vocabulary_is_sourced(self):
        text = "Dismissal is automatically unfair under section 103A if the principal reason was a protected disclosure by the employee."
        spans = confidence.provenance_spans(text, self.SOURCES)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].tier, "SOURCED")

    def test_unrelated_sentence_is_unsourced(self):
        text = "I personally think the claimant has a reasonably strong case on the facts described."
        spans = confidence.provenance_spans(text, self.SOURCES)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].tier, "UNSOURCED")

    def test_no_sources_at_all_everything_is_unsourced(self):
        # The entire point: an answer with nothing retrieved still gets a tier on every
        # sentence, unlike the old citation/attribution mechanisms which produced nothing.
        text = "Paris is the capital of France. It has a population of over two million."
        spans = confidence.provenance_spans(text, [])
        self.assertEqual(len(spans), 2)
        self.assertTrue(all(s.tier == "UNSOURCED" for s in spans))

    def test_every_sentence_gets_a_tier_none_omitted(self):
        text = "## Summary\n\nFirst point here about something unrelated. Second point also unrelated."
        spans = confidence.provenance_spans(text, self.SOURCES)
        # Heading + 2 sentences = 3 candidates, all three must appear.
        self.assertEqual(len(spans), 3)

    def test_sentence_containing_citation_anchor_is_skipped(self):
        # Left to quoted_citation_spans (href-matched) instead — a sentence spanning into an
        # embedded <a> cannot be matched as one contiguous text node after rendering.
        text = 'See <a href="https://x.gov.uk/s1">Some Act Law</a> for the rule, which applies broadly.'
        self.assertEqual(confidence._candidate_sentences(text), [])

    def test_ordered_list_marker_is_stripped(self):
        # marked renders "1. " as <ol> numbering, not text -- keeping it made the quote unmatchable.
        sentences = confidence._candidate_sentences("1. The employee must have two years' service.\n2) Then claim.")
        self.assertIn("The employee must have two years' service.", sentences)
        self.assertIn("Then claim.", sentences)

    def test_table_row_splits_into_matchable_cells(self):
        text = "| Event | Date |\n| --- | --- |\n| Dismissal | 1 March |"
        sentences = confidence._candidate_sentences(text)
        self.assertIn("Event", sentences)
        self.assertIn("Dismissal", sentences)
        self.assertIn("1 March", sentences)
        self.assertNotIn("---", sentences)

    def test_quotes_match_what_marked_parse_actually_renders(self):
        text = "## Summary\n\nThe **cap** applies here but is not drawn from any source."
        spans = confidence.provenance_spans(text, [])
        quotes = [s.quote for s in spans]
        self.assertIn("Summary", quotes)  # "##" stripped
        self.assertTrue(any("The cap applies here" in q for q in quotes))  # "**" stripped

    def test_empty_text_returns_nothing(self):
        self.assertEqual(confidence.provenance_spans("", self.SOURCES), [])

    def test_long_enough_substring_is_quoted(self):
        text = "The employer agreed to settle the claim in full within fourteen days of the hearing."
        spans = confidence.provenance_spans(text, [text])
        self.assertEqual(spans[0].tier, "QUOTED")

    def test_short_sentence_is_never_quoted_even_if_substring_coincidence(self):
        # QUOTE_MIN_CHARS guards against a short, generic phrase registering as a "quotation"
        # purely by coincidental overlap.
        short_spans = confidence.provenance_spans("It is.", ["It is a well established principle of law."])
        self.assertNotEqual(short_spans[0].tier, "QUOTED")


class QuotedCitationSpansTests(unittest.TestCase):
    POOL = [{"id": "abc", "title": "Manalo v. Republic", "url": "https://juris.ph/case/abc"}]
    ANCHOR_OK = '<a href="https://juris.ph/case/abc" class="legal-ref jurisprudence">Manalo Jurisprudence</a>'
    ANCHOR_BAD = '<a href="https://juris.ph/case/zzz-made-up" class="legal-ref jurisprudence">Fake Jurisprudence</a>'

    def test_resolved_link_is_quoted(self):
        spans = confidence.quoted_citation_spans(f"See {self.ANCHOR_OK}.", self.POOL)
        self.assertEqual(len(spans), 1)
        self.assertEqual(spans[0].tier, "QUOTED")
        self.assertEqual(spans[0].ref, "https://juris.ph/case/abc")

    def test_unresolved_link_produces_no_span(self):
        # Exactly three tiers now, no separate "flagged as wrong" case — an unresolved link
        # simply gets nothing here (its surrounding sentence, if any, falls to provenance_spans).
        spans = confidence.quoted_citation_spans(f"See {self.ANCHOR_BAD}.", self.POOL)
        self.assertEqual(spans, [])

    def test_empty_text_returns_nothing(self):
        self.assertEqual(confidence.quoted_citation_spans("", self.POOL), [])

    def test_fails_open_on_bad_search_results(self):
        # Must never raise regardless of what search_results looks like.
        spans = confidence.quoted_citation_spans(f"See {self.ANCHOR_OK}.", None)
        self.assertEqual(spans, [])


# ---------------------------------------------------------------------------
# model_registry.supports_logprobs
# ---------------------------------------------------------------------------

class SupportsLogprobsTests(unittest.TestCase):
    def test_capable_models(self):
        self.assertTrue(model_registry.supports_logprobs("gpt-4o-mini"))
        self.assertTrue(model_registry.supports_logprobs("gpt-4.1-mini"))
        self.assertTrue(model_registry.supports_logprobs("deepseek-chat"))

    def test_incapable_models(self):
        # The exact models this feature exists to work around — see confidence.py's docstring.
        self.assertFalse(model_registry.supports_logprobs("gpt-5.6-luna"))
        self.assertFalse(model_registry.supports_logprobs("gpt-5.6-terra"))
        self.assertFalse(model_registry.supports_logprobs("deepseek-reasoner"))

    def test_unknown_model_is_incapable_allowlist_not_denylist(self):
        self.assertFalse(model_registry.supports_logprobs("some-future-model-nobody-has-vetted"))
        self.assertFalse(model_registry.supports_logprobs(None))
        self.assertFalse(model_registry.supports_logprobs(""))

    def test_shape_chat_args_strips_logprobs_for_incapable_model(self):
        args = {"model": "gpt-5.6-luna", "logprobs": True, "top_logprobs": 5}
        out = model_registry.shape_chat_args("gpt-5.6-luna", args)
        self.assertNotIn("logprobs", out)
        self.assertNotIn("top_logprobs", out)

    def test_shape_chat_args_keeps_logprobs_for_capable_model(self):
        args = {"model": "gpt-4o-mini", "logprobs": True, "top_logprobs": 5}
        out = model_registry.shape_chat_args("gpt-4o-mini", args)
        self.assertIn("logprobs", out)


# ---------------------------------------------------------------------------
# Regression: the exact failure mode from the project handoff notes
# ---------------------------------------------------------------------------

def _logprob_tok(token_text, logprobs):
    return SimpleNamespace(token=token_text, top_logprobs=[SimpleNamespace(logprob=lp) for lp in logprobs])


def _cc_stream_with_logprobs(words_and_logprobs):
    """One SimpleNamespace chunk per word, each carrying a one-token logprobs.content list —
    the shape the_server.py's streaming loop reads via chunk.choices[0].logprobs.content."""
    chunks = []
    for text, logprobs in words_and_logprobs:
        delta = SimpleNamespace(content=text, tool_calls=None, function_call=None)
        lp = SimpleNamespace(content=[_logprob_tok(text, logprobs)])
        chunks.append(SimpleNamespace(choices=[SimpleNamespace(delta=delta, logprobs=lp)]))
    return chunks


class _FakeClient:
    def __init__(self, chunks):
        self._chunks = chunks
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return list(self._chunks)


def _collect(agen):
    async def run():
        out = []
        async for chunk in agen:
            out.append(chunk)
        return out

    return asyncio.run(run())


class ChatWonderStreamingTests(unittest.TestCase):
    """Drives the real streaming_run_function_chain (not a reimplementation of it) with fake
    logprobs, against a model_registry patch that makes it "telemetry-capable" regardless of
    which model name the test happens to use."""

    WORDS = [(" The", [-0.1, -3.0, -4.0]), (" answer", [-0.05, -5.0]), (" is", [-0.01, -8.0]), (" clear.", [-0.02])]

    def _run(self, confidence_live, confidence_spans_env="true"):
        chunks = _cc_stream_with_logprobs(self.WORDS)
        client = _FakeClient(chunks)
        state = srv.ChatState()
        state.openai_client = client
        state.confidence_live = confidence_live
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        with patch.dict(os.environ, {"CONFIDENCE_SPANS": confidence_spans_env}):
            with patch.object(model_registry, "supports_logprobs", return_value=True):
                out = _collect(srv.streaming_run_function_chain(state, msgs, max_chains=5, tools=[], auto_approval=True))
        return out, state, client

    def test_conf_frames_never_reach_full_response(self):
        """The regression for the documented prior failure: [METRIC] frames with no `continue`
        guard accumulated into full_response, and the legal path's single final send() then
        contained hundreds of interleaved tags — JSON.parse on the client read from the first
        opening tag to the last closing tag across the WHOLE response, failed silently, and the
        connection dropped mid-send. This asserts the generator itself never conflates the two:
        every [CONF] frame is a standalone yield, and the text-only reconstruction (what
        chat_stream's websocket loop would accumulate into full_response) contains no [CONF]."""
        out, state, client = self._run(confidence_live=True)
        conf_frames = [c for c in out if c.startswith("[CONF]")]
        self.assertTrue(len(conf_frames) > 0, "expected live [CONF] frames when confidence_live=True")
        for frame in conf_frames:
            self.assertTrue(frame.endswith("[/CONF]"))
            json.loads(frame[len("[CONF]"):-len("[/CONF]")])  # must be valid, self-contained JSON
        full_response = "".join(c for c in out if not c.startswith("[CONF]") and not c.startswith("[TRACE]"))
        self.assertNotIn("[CONF]", full_response)
        self.assertEqual(full_response, "".join(w for w, _ in self.WORDS))

    def test_no_live_frames_when_confidence_live_false(self):
        out, state, client = self._run(confidence_live=False)
        self.assertEqual([c for c in out if c.startswith("[CONF]")], [])
        # Entropy is still captured for the post-__END__ envelope even without live frames.
        self.assertTrue(len(state.confidence_tokens) > 0)

    def test_confidence_tokens_built_from_logprobs_token_not_delta_text(self):
        out, state, client = self._run(confidence_live=False)
        token_texts = [t for t, _h in state.confidence_tokens]
        self.assertEqual(token_texts, [w for w, _ in self.WORDS])

    def test_disabled_flag_requests_no_logprobs_and_emits_nothing(self):
        out, state, client = self._run(confidence_live=True, confidence_spans_env="false")
        self.assertEqual([c for c in out if c.startswith("[CONF]")], [])
        for call in client.calls:
            self.assertNotIn("logprobs", call)
            self.assertNotIn("top_logprobs", call)

    def test_incapable_model_requests_no_logprobs(self):
        """With the real (unpatched) supports_logprobs, gpt-5.6-luna must never get a logprobs
        request even if confidence_live is set — the request-field opt-in only takes effect on
        a model that can actually honor it."""
        chunks = _cc_stream_with_logprobs(self.WORDS)
        client = _FakeClient(chunks)
        state = srv.ChatState()
        state.openai_client = client
        state.confidence_live = True
        msgs = [{"role": "system", "content": "sys"}, {"role": "user", "content": "q"}]
        out = _collect(srv.streaming_run_function_chain(
            state, msgs, max_chains=5, tools=[], auto_approval=True, model="gpt-5.6-luna",
        ))
        self.assertEqual([c for c in out if c.startswith("[CONF]")], [])
        for call in client.calls:
            self.assertNotIn("logprobs", call)

    def test_state_generated_never_contains_conf(self):
        """End-to-end check of ChatState.__getstate__'s hygiene: confidence_tokens must not
        survive a pickle, matching the per-turn-only lifetime the plan specifies."""
        out, state, client = self._run(confidence_live=True)
        pickled_state = state.__getstate__()
        self.assertIsNone(pickled_state["confidence_tokens"])


class LongAnswerCoverageTests(unittest.TestCase):
    """Found live (2026-11-10): a 31K-char UK answer had 407 text blocks but only the first 80
    sentences were ever scored, leaving everything below the first screenful unshaded -- which
    reads as "never checked", the exact failure _candidate_sentences' docstring forbids."""

    def test_every_sentence_of_a_long_answer_gets_a_span(self):
        text = "\n".join(f"- Point number {i} about unfair dismissal." for i in range(300))
        spans = confidence.provenance_spans(text, ["unfair dismissal rules"])
        self.assertEqual(len(spans), 300)


class ConfidenceSourcesTests(unittest.TestCase):
    """Found live (2026-11-10): 67 of 71 spans on a UK answer were UNSOURCED because the corpus
    was only search-row title+snippet ("The right.", "General."), while the model had actually
    read full statute text via legislation_get_section -- text already sitting in the pool."""

    def _state(self, pool):
        return SimpleNamespace(last_search_legal_results=pool, case_documents=None, prompt=[])

    def test_uk_section_full_text_reaches_the_corpus(self):
        section_text = ("An employee who is dismissed shall be regarded as unfairly dismissed if the "
                        "reason for the dismissal is that the employee made a protected disclosure.")
        pool = [{"title": "General.", "url": "https://www.legislation.gov.uk/ukpga/1996/18/section/103A",
                 "section": {"heading": "Protected disclosure", "content": section_text}}]
        with patch.object(srv, "exhibit_texts", return_value=[]):
            sources = srv._confidence_sources("legal_uk", self._state(pool))
        self.assertEqual(len(sources), 1)  # one document per pool row, so IDF stays meaningful
        self.assertIn("protected disclosure", sources[0])

    def test_ph_get_case_reasoning_reaches_the_corpus(self):
        pool = [{"title": "People v. X", "snippet": "", "court_reasoning":
                 "The constitutional right against unreasonable searches cannot be waived by silence."}]
        with patch.object(srv, "exhibit_texts", return_value=[]):
            sources = srv._confidence_sources("legal", self._state(pool))
        self.assertTrue(any("unreasonable searches" in s for s in sources))

    def test_sentence_drawn_from_full_section_text_is_not_unsourced(self):
        section_text = ("An employee who is dismissed shall be regarded as unfairly dismissed if the "
                        "reason for the dismissal is that the employee made a protected disclosure.")
        pool = [{"title": "General.", "section": {"content": section_text}},
                {"title": "Qualifying period of employment.", "snippet": "two years continuous employment"}]
        with patch.object(srv, "exhibit_texts", return_value=[]):
            sources = srv._confidence_sources("legal_uk", self._state(pool))
        spans = confidence.provenance_spans(
            "Dismissal because the employee made a protected disclosure is automatically unfair.", sources)
        self.assertNotEqual(spans[0].tier, "UNSOURCED")

    def test_verbatim_match_survives_curly_quotes(self):
        # The pool corpus strips quote characters (legal_citations._normalize_quote_text); the
        # sentence side must be normalized the same way or a real quotation never matches.
        src = "The employer shall show the reason, or if more than one the principal reason, for the dismissal."
        spans = confidence.provenance_spans(
            "The employer shall show “the reason, or if more than one the principal reason”, for the dismissal.",
            [src.lower()])
        self.assertEqual(spans[0].tier, "QUOTED")


if __name__ == "__main__":
    unittest.main()
