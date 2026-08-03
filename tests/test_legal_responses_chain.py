"""Unit tests for the Responses-API tool/input conversion helpers."""

import unittest

from legal_responses_chain import (
    append_function_call_round_trip,
    chat_messages_to_input_items,
    to_responses_tools,
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


if __name__ == "__main__":
    unittest.main()
