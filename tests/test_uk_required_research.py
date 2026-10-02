"""The UK legal persona must research before it answers a real legal question, whatever model is configured.

Live, in ~15 turns the model made no legislation search at all: it answered from memory and cited nothing, and
which model/effort was set changed how often. The first model call of a qualifying turn now requires the UK
research tool (tool_choice = that function); later calls go back to "auto". The OpenAI client is faked."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import legal_responses_chain as lrc
import the_server

TOOL = "get_legal_recommendation_uk"
MANIFEST = [
    {"type": "function", "function": {"name": TOOL, "description": "d", "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {"name": "legislation_search", "description": "d", "parameters": {"type": "object", "properties": {}}}},
]
QUESTION = "I need to fix my birth certificate because it has a different name from what I have used. What are my options?"


def _state(**attrs):
    return SimpleNamespace(**attrs)


class RuleTests(unittest.TestCase):
    def test_a_real_uk_legal_question_requires_the_research_tool(self):
        self.assertEqual(the_server._uk_forced_first_tool("legal_uk", _state(), QUESTION, MANIFEST), TOOL)

    def test_other_personas_are_untouched(self):
        for persona in ("legal", "auto", "garment"):
            self.assertIsNone(the_server._uk_forced_first_tool(persona, _state(), QUESTION, MANIFEST))

    def test_a_short_message_is_not_forced(self):
        self.assertIsNone(the_server._uk_forced_first_tool("legal_uk", _state(), "thanks, that helps", MANIFEST))

    def test_structured_one_shot_calls_opt_out_with_skip_legal_verify(self):
        self.assertIsNone(the_server._uk_forced_first_tool("legal_uk", _state(skip_legal_verify=True), QUESTION, MANIFEST))

    def test_the_caller_can_opt_out_per_request(self):
        self.assertIsNone(the_server._uk_forced_first_tool("legal_uk", _state(require_research=False), QUESTION, MANIFEST))
        self.assertEqual(the_server._uk_forced_first_tool("legal_uk", _state(require_research=True), QUESTION, MANIFEST), TOOL)

    def test_it_can_be_switched_off_for_the_whole_server(self):
        with patch.dict(os.environ, {"UK_REQUIRE_RESEARCH": "false"}):
            self.assertIsNone(the_server._uk_forced_first_tool("legal_uk", _state(), QUESTION, MANIFEST))

    def test_nothing_is_forced_if_the_tool_is_not_offered(self):
        only_search = [MANIFEST[1]]
        self.assertIsNone(the_server._uk_forced_first_tool("legal_uk", _state(), QUESTION, only_search))
        self.assertIsNone(the_server._uk_forced_first_tool("legal_uk", _state(), QUESTION, None))


def _call(name):
    item = SimpleNamespace(type="function_call", call_id="call-1", name=name, arguments='{"legal_issue": "birth certificate name"}')
    return [
        SimpleNamespace(type="response.output_item.added", output_index=0, item=item),
        SimpleNamespace(type="response.output_item.done", output_index=0, item=item),
    ]


def _text(text):
    return [SimpleNamespace(type="response.output_text.delta", delta=text)]


class FakeOpenAI:
    """Replies to successive responses.create calls from `script` and records every call's arguments."""

    def __init__(self, script):
        self.script, self.calls = list(script), []
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **args):
        self.calls.append(args)
        return iter(self.script[len(self.calls) - 1])


class ChainTests(unittest.TestCase):
    def _run(self, client, **kwargs):
        state = SimpleNamespace(openai_client=client, last_search_legal_results=[], turn_tool_calls=0)
        messages = [{"role": "user", "content": QUESTION}]
        with patch.object(the_server, "execute_function_call", return_value={"success": True, "legislation": [], "case_law": []}):
            return lrc.run_function_chain_responses(
                state, messages, session_id=None, tools=MANIFEST, query=QUESTION, model="m", reasoning_effort="low",
                auto_approval=True, **kwargs,
            )

    def _choices(self, client):
        return [c["tool_choice"] for c in client.calls]

    def test_only_the_first_call_is_forced_and_the_model_still_answers_after_it(self):
        client = FakeOpenAI([_call(TOOL), _text("Answer after research.")])
        self._run(client, force_first_tool=TOOL)
        self.assertEqual(self._choices(client), [{"type": "function", "name": TOOL}, "auto"])

    def test_without_the_option_every_call_is_auto(self):
        client = FakeOpenAI([_text("Answer.")])
        self._run(client)
        self.assertEqual(self._choices(client), ["auto"])

    def test_a_tool_that_is_not_offered_is_not_forced(self):
        client = FakeOpenAI([_text("Answer.")])
        self._run(client, force_first_tool="not_a_tool")
        self.assertEqual(self._choices(client), ["auto"])


if __name__ == "__main__":
    unittest.main()
