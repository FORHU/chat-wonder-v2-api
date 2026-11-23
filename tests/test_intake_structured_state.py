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
