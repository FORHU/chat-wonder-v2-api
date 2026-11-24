# tests/test_intake_structured_state.py
"""#158: the [INTAKE_SUGGESTIONS] payload the websocket handler sends for INTAKE_PERSONAS turns.
Wiring-only tests against the pure helper functions, same mocking style as test_describe_image.py
-- no real OpenAI call, no live data.

#157's field/missing mapping used to live here too (as [INTAKE_STATE]) but was removed: it
duplicated ilovelawyer-api's existing ClientIntakeSvc extraction, which derives the same fields
from data it already has with no second LLM call. See the plan's reconciliation note."""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import the_server as srv


def _fake_json_client(payload_json: str):
    def create(**kw):
        return NS(choices=[NS(message=NS(content=payload_json), finish_reason="stop")])
    return NS(chat=NS(completions=NS(create=create)))


class IntakeConversationTextTests(unittest.TestCase):
    def test_conversation_text_interleaves_client_and_assistant_turns(self):
        state = srv.ChatState()
        state.prompt = ["my landlord won't fix the boiler"]
        state.generated = ["Where are you based?"]
        text = srv._intake_conversation_text(state)
        self.assertIn("Client: my landlord won't fix the boiler", text)
        self.assertIn("Assistant: Where are you based?", text)


class GenerateIntakeSuggestionsTests(unittest.TestCase):
    def test_returns_empty_list_when_the_assistant_has_not_spoken_yet(self):
        state = srv.ChatState()
        self.assertEqual(srv._generate_intake_suggestions(state), [])

    def test_returns_suggestions_capped_at_four(self):
        state = srv.ChatState()
        state.prompt = ["my landlord won't fix the boiler"]
        state.generated = ["When did you first report it?"]
        payload = '{"suggestions": ["Last week", "Last month", "Over a month ago", "I have not reported it yet", "Extra one"]}'
        state.openai_client = _fake_json_client(payload)
        suggestions = srv._generate_intake_suggestions(state)
        self.assertEqual(len(suggestions), 4)
        self.assertIn("Last week", suggestions)

    def test_empty_suggestions_list_is_returned_as_is_not_an_error(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["What's going on?"]
        state.openai_client = _fake_json_client('{"suggestions": []}')
        self.assertEqual(srv._generate_intake_suggestions(state), [])

    def test_returns_empty_list_on_openai_failure_rather_than_raising(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["What's going on?"]

        def raise_err(**kw):
            raise RuntimeError("boom")

        state.openai_client = NS(chat=NS(completions=NS(create=raise_err)))
        self.assertEqual(srv._generate_intake_suggestions(state), [])

    def test_the_jurisdiction_question_always_gets_the_real_three_values_verbatim(self):
        """Live QA (2026-11-23) caught the model splitting "England and Wales" into two separate,
        invalid suggestions. This question's answer set is known in advance, so it's short-circuited
        before any OpenAI call is even made -- never trusted to a free-form rephrasing."""
        state = srv.ChatState()
        state.prompt = ["I was dismissed from my job."]
        state.generated = ["Which part of the UK did this happen in — England and Wales, Scotland, or Northern Ireland — and roughly where?"]

        def fail_if_called(**kw):
            raise AssertionError("should have short-circuited before calling OpenAI")

        state.openai_client = NS(chat=NS(completions=NS(create=fail_if_called)))
        self.assertEqual(srv._generate_intake_suggestions(state), ["England and Wales", "Scotland", "Northern Ireland"])

    def test_the_suggestions_prompt_refuses_a_compound_question(self):
        """2026-11-24: live QA caught CW asking "roughly when this happened, and where in the
        UK you are?" -- not the dedicated jurisdiction question (so the marker short-circuit
        above doesn't apply), but still two distinct facts bundled into one question. The
        suggestions this produced mixed date options and a place option in one flat list, with
        no way to answer both -- the suggestions prompt must refuse to suggest anything for a
        question shaped like this, same principle as never splitting a compound option."""
        prompt = srv._intake_suggestions_prompt("irrelevant for this check")
        self.assertIn("bundles two distinct asks together", prompt)
        self.assertIn("more than one distinct fact", prompt)

    def test_the_suggestions_prompt_forbids_inventing_a_name_nobody_gave(self):
        """2026-11-24: live QA caught suggestions inventing four plausible-sounding but entirely
        fictional moving-company names ("ABC Movers", "XYZ Relocation Services", ...) for "what's
        the name of the moving company you booked?" -- nobody had named a company anywhere in the
        conversation, so every suggestion was fabricated; tapping one would have written a made-up
        company name straight into the case record."""
        prompt = srv._intake_suggestions_prompt("irrelevant for this check")
        self.assertIn("hasn't been named yet anywhere in the conversation", prompt)
        self.assertIn("ABC Movers", prompt)
        self.assertIn("not taking their situation seriously", prompt)
        self.assertIn("return an empty list for this shape of question", prompt)


class BuildIntakeSuggestionsPayloadTests(unittest.TestCase):
    def test_returns_none_when_the_assistant_has_not_spoken_yet(self):
        state = srv.ChatState()
        self.assertIsNone(srv._build_intake_suggestions_payload(state))

    def test_wraps_suggestions_in_the_payload_shape(self):
        state = srv.ChatState()
        state.prompt = ["my landlord won't fix the boiler"]
        state.generated = ["When did you first report it?"]
        state.openai_client = _fake_json_client('{"suggestions": ["Last week", "Last month"]}')
        payload = srv._build_intake_suggestions_payload(state)
        self.assertEqual(payload, {"suggestions": ["Last week", "Last month"]})

    def test_sends_an_explicit_empty_list_rather_than_none_when_nothing_is_confident(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["What's going on?"]
        state.openai_client = _fake_json_client('{"suggestions": []}')
        payload = srv._build_intake_suggestions_payload(state)
        self.assertEqual(payload, {"suggestions": []})


if __name__ == "__main__":
    unittest.main()
