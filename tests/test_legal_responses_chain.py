"""Unit tests for the Responses-API tool/input conversion helpers."""

import unittest
from types import SimpleNamespace as NS

from legal_responses_chain import (
    append_function_call_round_trip,
    apply_responses_event,
    chat_messages_to_input_items,
    items_to_chat_messages,
    to_responses_tools,
    _deepseek_completions_events,
)


class ToResponsesToolsTests(unittest.TestCase):
    def test_flattens_chat_completions_shape(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "search_jurisprudence",
                    "description": "Search cases",
                    "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
                },
            }
        ]
        out = to_responses_tools(tools)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["type"], "function")
        self.assertEqual(out[0]["name"], "search_jurisprudence")
        self.assertEqual(out[0]["description"], "Search cases")
        self.assertIn("parameters", out[0])
        self.assertNotIn("function", out[0])

    def test_empty_or_none_returns_empty_list(self):
        self.assertEqual(to_responses_tools(None), [])
        self.assertEqual(to_responses_tools([]), [])

    def test_skips_malformed_entries(self):
        out = to_responses_tools([{"type": "function", "function": {}}, {"garbage": True}])
        self.assertEqual(out, [])

    def test_missing_description_defaults_to_empty_string(self):
        tools = [{"type": "function", "function": {"name": "foo"}}]
        out = to_responses_tools(tools)
        self.assertEqual(out[0]["description"], "")
        self.assertEqual(out[0]["parameters"], {"type": "object", "properties": {}})


class ChatMessagesToInputItemsTests(unittest.TestCase):
    def test_passthrough_role_content_pairs(self):
        messages = [
            {"role": "system", "content": "sys prompt"},
            {"role": "user", "content": "hello"},
        ]
        out = chat_messages_to_input_items(messages)
        self.assertEqual(out, messages)

    def test_returns_new_list_not_same_object(self):
        messages = [{"role": "user", "content": "hi"}]
        out = chat_messages_to_input_items(messages)
        self.assertIsNot(out, messages)
        out.append({"role": "user", "content": "mutated"})
        self.assertEqual(len(messages), 1)


class AppendFunctionCallRoundTripTests(unittest.TestCase):
    def test_appends_call_and_output_items(self):
        items = [{"role": "user", "content": "q"}]
        result = {"success": True, "results": [1, 2, 3]}
        out = append_function_call_round_trip(
            items, call_id="call_123", name="search_jurisprudence", arguments='{"query":"x"}', result=result
        )
        self.assertIs(out, items)
        self.assertEqual(len(items), 3)
        self.assertEqual(items[1], {"type": "function_call", "call_id": "call_123", "name": "search_jurisprudence", "arguments": '{"query":"x"}'})
        self.assertEqual(items[2]["type"], "function_call_output")
        self.assertEqual(items[2]["call_id"], "call_123")
        self.assertIn('"success": true', items[2]["output"])

    def test_string_result_passed_through_unwrapped(self):
        items = []
        append_function_call_round_trip(items, call_id="c1", name="f", arguments="{}", result="already a string")
        self.assertEqual(items[1]["output"], "already a string")

    def test_unserializable_result_falls_back_to_str(self):
        class Weird:
            def __str__(self):
                return "weird-repr"

        items = []
        append_function_call_round_trip(items, call_id="c1", name="f", arguments="{}", result=Weird())
        self.assertEqual(items[1]["output"], "weird-repr")


class ItemsToChatMessagesTests(unittest.TestCase):
    """CW-7's DeepSeek bridge: a one-way, call-time-only translation of this module's
    Responses-shaped input_items into Chat Completions messages."""

    def test_passthrough_role_content_items(self):
        items = [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        self.assertEqual(items_to_chat_messages(items), items)

    def test_function_call_and_output_pair_becomes_assistant_and_tool_messages(self):
        items = [{"role": "user", "content": "q"}]
        append_function_call_round_trip(
            items, call_id="call_1", name="search_jurisprudence", arguments='{"query":"x"}', result={"ok": True}
        )
        out = items_to_chat_messages(items)
        self.assertEqual(len(out), 3)
        self.assertEqual(out[0], {"role": "user", "content": "q"})
        self.assertEqual(out[1]["role"], "assistant")
        self.assertEqual(out[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(out[1]["tool_calls"][0]["function"]["name"], "search_jurisprudence")
        self.assertEqual(out[1]["tool_calls"][0]["function"]["arguments"], '{"query":"x"}')
        self.assertEqual(out[2], {"role": "tool", "tool_call_id": "call_1", "content": '{"ok": true}'})

    def test_multiple_pairs_each_become_their_own_assistant_tool_pair(self):
        items = []
        append_function_call_round_trip(items, call_id="c1", name="f1", arguments="{}", result="r1")
        append_function_call_round_trip(items, call_id="c2", name="f2", arguments="{}", result="r2")
        out = items_to_chat_messages(items)
        self.assertEqual(len(out), 4)
        self.assertEqual([m["role"] for m in out], ["assistant", "tool", "assistant", "tool"])
        self.assertEqual(out[1]["tool_call_id"], "c1")
        self.assertEqual(out[3]["tool_call_id"], "c2")

    def test_function_call_output_with_no_matching_call_is_dropped_not_raised(self):
        items = [{"type": "function_call_output", "call_id": "orphan", "output": "x"}]
        self.assertEqual(items_to_chat_messages(items), [])


class FakeDeepseekCompletions:
    def __init__(self, stream_chunks):
        self.kwargs = None
        self._stream_chunks = stream_chunks

    def create(self, **kw):
        self.kwargs = kw
        return iter(self._stream_chunks)


class FakeDeepseekClient:
    def __init__(self, stream_chunks):
        self.chat = NS(completions=FakeDeepseekCompletions(stream_chunks))


def _tc_delta(index, call_id=None, name=None, arguments=None):
    fn = NS(name=name, arguments=arguments)
    return NS(index=index, id=call_id, function=fn)


class DeepseekCompletionsEventsTests(unittest.TestCase):
    """Shapes confirmed live against api.deepseek.com before writing this: id/name arrive only
    on a tool call's first delta chunk; every later chunk for it carries only an arguments
    piece; no closing event is ever sent."""

    def test_tool_call_deltas_reconstruct_into_pending_calls_via_apply_responses_event(self):
        chunks = [
            NS(choices=[NS(delta=NS(content="", tool_calls=None))]),
            NS(choices=[NS(delta=NS(content=None, tool_calls=[_tc_delta(0, call_id="call_00_x", name="get_weather", arguments="")]))]),
            NS(choices=[NS(delta=NS(content=None, tool_calls=[_tc_delta(0, arguments='{"')]))]),
            NS(choices=[NS(delta=NS(content=None, tool_calls=[_tc_delta(0, arguments='city": "Paris"}')]))]),
            NS(choices=[NS(delta=NS(content="", tool_calls=None))]),
        ]
        client = FakeDeepseekClient(chunks)
        pending_calls = {}
        for event in _deepseek_completions_events(client, "deepseek-chat", [{"role": "user", "content": "q"}], None):
            apply_responses_event(event, pending_calls)
        self.assertEqual(pending_calls, {0: {"call_id": "call_00_x", "name": "get_weather", "arguments": '{"city": "Paris"}'}})

    def test_text_deltas_are_yielded_as_output_text_delta_events(self):
        chunks = [
            NS(choices=[NS(delta=NS(content="Hello", tool_calls=None))]),
            NS(choices=[NS(delta=NS(content=" there", tool_calls=None))]),
        ]
        client = FakeDeepseekClient(chunks)
        text = ""
        for event in _deepseek_completions_events(client, "deepseek-chat", [{"role": "user", "content": "q"}], None):
            delta = apply_responses_event(event, {})
            if delta:
                text += delta
        self.assertEqual(text, "Hello there")

    def test_a_chunk_with_no_choices_is_skipped_not_raised(self):
        chunks = [NS(choices=[]), NS(choices=[NS(delta=NS(content="ok", tool_calls=None))])]
        client = FakeDeepseekClient(chunks)
        events = list(_deepseek_completions_events(client, "deepseek-chat", [], None))
        self.assertEqual(len(events), 1)

    def test_shapes_chat_args_through_model_registry_so_deepseek_gets_thinking_disabled(self):
        client = FakeDeepseekClient([NS(choices=[NS(delta=NS(content="ok", tool_calls=None))])])
        list(_deepseek_completions_events(client, "deepseek-chat", [{"role": "user", "content": "q"}], None))
        kwargs = client.chat.completions.kwargs
        self.assertEqual(kwargs["extra_body"]["thinking"], {"type": "disabled"})
        self.assertNotIn("tools", kwargs)

    def test_tools_are_passed_through_when_provided(self):
        tools = [{"type": "function", "function": {"name": "get_weather", "parameters": {}}}]
        client = FakeDeepseekClient([NS(choices=[NS(delta=NS(content="ok", tool_calls=None))])])
        list(_deepseek_completions_events(client, "deepseek-chat", [{"role": "user", "content": "q"}], tools))
        kwargs = client.chat.completions.kwargs
        self.assertEqual(kwargs["tools"], tools)
        self.assertEqual(kwargs["tool_choice"], "auto")
        self.assertFalse(kwargs["parallel_tool_calls"])


if __name__ == "__main__":
    unittest.main()
