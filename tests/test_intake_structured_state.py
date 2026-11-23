# tests/test_intake_structured_state.py
"""#157/#158: the [INTAKE_STATE] payload (fields/missing/suggestions) the websocket handler sends
for INTAKE_PERSONAS turns. Wiring-only tests against the pure helper functions, same mocking
style as test_describe_image.py -- no real OpenAI call, no live data."""
import unittest
from types import SimpleNamespace as NS
from unittest import mock

import the_server as srv


def _fake_json_client(payload_json: str):
    def create(**kw):
        return NS(choices=[NS(message=NS(content=payload_json), finish_reason="stop")])
    return NS(chat=NS(completions=NS(create=create)))


class IntakeConversationHelpersTests(unittest.TestCase):
    def test_fingerprint_changes_when_a_turn_is_added(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["hi there"]
        fp1 = srv._intake_conversation_fingerprint(state)
        state.prompt.append("my landlord won't fix the boiler")
        state.generated.append("I'm sorry to hear that. Where are you based?")
        fp2 = srv._intake_conversation_fingerprint(state)
        self.assertNotEqual(fp1, fp2)

    def test_fingerprint_is_stable_for_the_same_conversation(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["hi there"]
        self.assertEqual(
            srv._intake_conversation_fingerprint(state),
            srv._intake_conversation_fingerprint(state),
        )

    def test_conversation_text_interleaves_client_and_assistant_turns(self):
        state = srv.ChatState()
        state.prompt = ["my landlord won't fix the boiler"]
        state.generated = ["Where are you based?"]
        text = srv._intake_conversation_text(state)
        self.assertIn("Client: my landlord won't fix the boiler", text)
        self.assertIn("Assistant: Where are you based?", text)


class ExtractIntakeFieldsTests(unittest.TestCase):
    def test_returns_none_for_an_empty_conversation(self):
        state = srv.ChatState()
        self.assertIsNone(srv._extract_intake_fields(state))

    def test_splits_every_form_field_into_fields_or_missing_never_both(self):
        state = srv.ChatState()
        state.prompt = ["My landlord in Manchester won't fix the boiler and I want it repaired."]
        state.generated = ["Thanks — when did you first report it?"]
        payload = (
            '{"fields": {"parties": "client and landlord", "venue": "Manchester", '
            '"client_goals": "wants the boiler repaired"}, "missing": []}'
        )
        state.openai_client = _fake_json_client(payload)
        result = srv._extract_intake_fields(state)
        self.assertEqual(set(result["fields"]) | set(result["missing"]), set(srv.INTAKE_FORM_FIELDS))
        self.assertEqual(set(result["fields"]) & set(result["missing"]), set())
        self.assertEqual(result["fields"]["venue"], "Manchester")
        self.assertIn("type_of_action", result["missing"])

    def test_drops_a_field_name_the_model_invented(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["hi"]
        payload = '{"fields": {"parties": "x", "not_a_real_field": "y"}, "missing": []}'
        state.openai_client = _fake_json_client(payload)
        result = srv._extract_intake_fields(state)
        self.assertNotIn("not_a_real_field", result["fields"])

    def test_caches_by_fingerprint_so_an_unchanged_conversation_skips_the_call(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["hi"]
        calls = []

        def create(**kw):
            calls.append(1)
            return NS(choices=[NS(message=NS(content='{"fields": {}, "missing": []}'), finish_reason="stop")])

        state.openai_client = NS(chat=NS(completions=NS(create=create)))
        srv._extract_intake_fields(state)
        srv._extract_intake_fields(state)
        self.assertEqual(len(calls), 1)

    def test_cache_misses_once_a_new_turn_is_added(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["hi"]
        calls = []

        def create(**kw):
            calls.append(1)
            return NS(choices=[NS(message=NS(content='{"fields": {}, "missing": []}'), finish_reason="stop")])

        state.openai_client = NS(chat=NS(completions=NS(create=create)))
        srv._extract_intake_fields(state)
        state.prompt.append("more detail")
        state.generated.append("thanks")
        srv._extract_intake_fields(state)
        self.assertEqual(len(calls), 2)

    def test_returns_none_on_openai_failure_rather_than_raising(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["hi"]

        def raise_err(**kw):
            raise RuntimeError("boom")

        state.openai_client = NS(chat=NS(completions=NS(create=raise_err)))
        self.assertIsNone(srv._extract_intake_fields(state))


class GenerateIntakeSuggestionsTests(unittest.TestCase):
    def test_returns_empty_list_when_the_assistant_has_not_spoken_yet(self):
        state = srv.ChatState()
        self.assertEqual(srv._generate_intake_suggestions(state, None), [])

    def test_returns_suggestions_capped_at_four(self):
        state = srv.ChatState()
        state.prompt = ["my landlord won't fix the boiler"]
        state.generated = ["When did you first report it?"]
        payload = '{"suggestions": ["Last week", "Last month", "Over a month ago", "I have not reported it yet", "Extra one"]}'
        state.openai_client = _fake_json_client(payload)
        suggestions = srv._generate_intake_suggestions(state, None)
        self.assertEqual(len(suggestions), 4)
        self.assertIn("Last week", suggestions)

    def test_empty_suggestions_list_is_returned_as_is_not_an_error(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["What's going on?"]
        state.openai_client = _fake_json_client('{"suggestions": []}')
        self.assertEqual(srv._generate_intake_suggestions(state, None), [])

    def test_returns_empty_list_on_openai_failure_rather_than_raising(self):
        state = srv.ChatState()
        state.prompt = ["hello"]
        state.generated = ["What's going on?"]

        def raise_err(**kw):
            raise RuntimeError("boom")

        state.openai_client = NS(chat=NS(completions=NS(create=raise_err)))
        self.assertEqual(srv._generate_intake_suggestions(state, None), [])


class BuildIntakeStatePayloadTests(unittest.TestCase):
    def test_combines_fields_missing_and_suggestions_into_one_payload(self):
        state = srv.ChatState()
        state.prompt = ["my landlord won't fix the boiler"]
        state.generated = ["When did you first report it?"]

        calls = {"n": 0}

        def create(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                content = '{"fields": {"venue": "Manchester"}, "missing": []}'
            else:
                content = '{"suggestions": ["Last week", "Last month"]}'
            return NS(choices=[NS(message=NS(content=content), finish_reason="stop")])

        state.openai_client = NS(chat=NS(completions=NS(create=create)))
        payload = srv._build_intake_state_payload(state)
        self.assertEqual(payload["fields"]["venue"], "Manchester")
        self.assertIn("type_of_action", payload["missing"])
        self.assertEqual(payload["suggestions"], ["Last week", "Last month"])

    def test_returns_none_when_fields_extraction_returns_none(self):
        state = srv.ChatState()
        self.assertIsNone(srv._build_intake_state_payload(state))


if __name__ == "__main__":
    unittest.main()
