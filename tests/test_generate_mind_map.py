"""the_server._generate_mind_map (the case strategy map, its own call since the map left
_generate_structured_data) and the ChatRequest fields that gate it, against a scripted fake OpenAI
client. No live OpenAI call.
"""

import json
import unittest
from types import SimpleNamespace

import the_server as srv

srv._load_user_functions(overwrite_globals=True)

TREE = {
    "id": "root",
    "label": "Cruz v. Reyes — unpaid loan",
    "isRoot": True,
    "children": [
        {
            "id": "legalBasis",
            "label": "Legal Basis",
            "children": [
                {
                    "id": "x",
                    "label": "Breach of the note",
                    "description": "Nothing was paid by 1 June.",
                    "children": [{"id": "y", "label": "Element: due date passed", "children": []}],
                }
            ],
        },
        {"id": "keyFacts", "label": "Key Facts", "children": []},
    ],
}

# Longer than the 2,500 chars the old map read, with a marker past that point.
LONG_ANSWER = ("The analysis. " * 300) + "MARKER-PAST-2500"


def _completion(content: str, finish_reason: str = "stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish_reason)])


class _FakeClient:
    """Returns `content`; optionally rejects the first call that passes response_format, like a
    provider without JSON mode."""

    def __init__(self, content: str, reject_json_mode: bool = False, raises: Exception | None = None):
        self.content = content
        self.reject_json_mode = reject_json_mode
        self.raises = raises
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.raises:
            raise self.raises
        if self.reject_json_mode and "response_format" in kwargs:
            raise ValueError("Invalid parameter: 'response_format' of type 'json_object' is not supported with this model.")
        return _completion(self.content)


def _state(client):
    state = srv.ChatState()
    state.openai_client = client
    return state


class GenerateMindMapTests(unittest.TestCase):
    def test_returns_the_tree_using_json_mode_and_a_token_budget(self):
        client = _FakeClient(json.dumps(TREE))
        out = srv._generate_mind_map(LONG_ANSWER, _state(client))
        self.assertEqual(out["label"], "Cruz v. Reyes — unpaid loan")
        call = client.calls[0]
        self.assertEqual(call["response_format"], {"type": "json_object"})
        self.assertEqual(call["max_tokens"], srv._MIND_MAP_MAX_TOKENS)

    def test_reads_the_whole_answer_and_the_case_summary(self):
        client = _FakeClient(json.dumps(TREE))
        srv._generate_mind_map(LONG_ANSWER, _state(client), "Findings:\n- LEGAL_ISSUE: Default on the note")
        prompt = client.calls[0]["messages"][1]["content"]
        self.assertIn("MARKER-PAST-2500", prompt)
        self.assertIn("CASE SUMMARY", prompt)
        self.assertIn("Default on the note", prompt)
        # The lawyer's edits in the current map outrank the fresh analysis.
        self.assertIn("lawyer's own changes to the current map, which always stand", prompt)
        # The five fixed first-level ids ilovelawyer-api keys on.
        for branch_id, _ in srv._MIND_MAP_BRANCHES:
            self.assertIn(f'"{branch_id}"', prompt)

    def test_leaves_the_case_summary_section_out_when_there_is_none(self):
        client = _FakeClient(json.dumps(TREE))
        srv._generate_mind_map("Short answer.", _state(client), "   ")
        self.assertNotIn("CASE SUMMARY", client.calls[0]["messages"][1]["content"])

    def test_retries_without_json_mode_when_the_provider_rejects_it(self):
        client = _FakeClient(json.dumps(TREE), reject_json_mode=True)
        out = srv._generate_mind_map("Answer.", _state(client))
        self.assertIsNotNone(out)
        self.assertEqual(len(client.calls), 2)
        self.assertNotIn("response_format", client.calls[1])

    def test_accepts_a_mindMap_wrapper_and_a_code_fence(self):
        fenced = "```json\n" + json.dumps({"mindMap": TREE}) + "\n```"
        out = srv._generate_mind_map("Answer.", _state(_FakeClient(fenced)))
        self.assertEqual(out["id"], "root")

    def test_returns_none_for_an_empty_or_unusable_tree(self):
        self.assertIsNone(srv._generate_mind_map("Answer.", _state(_FakeClient("{}"))))
        self.assertIsNone(srv._generate_mind_map("Answer.", _state(_FakeClient(json.dumps({**TREE, "children": []})))))
        self.assertIsNone(srv._generate_mind_map("Answer.", _state(_FakeClient("not json at all"))))

    def test_returns_none_instead_of_raising_when_the_call_fails(self):
        client = _FakeClient("", raises=RuntimeError("timeout"))
        self.assertIsNone(srv._generate_mind_map("Answer.", _state(client)))
        # Not a JSON-mode rejection, so no second attempt.
        self.assertEqual(len(client.calls), 1)

    def test_stats_count_nodes_and_depth(self):
        self.assertEqual(srv._mind_map_stats(TREE), (5, 3))


class StructuredDataTests(unittest.TestCase):
    def test_structured_data_no_longer_asks_for_a_mind_map(self):
        client = _FakeClient(json.dumps({"timeline": [{"title": "File", "description": "d", "status": "pending"}]}))
        out = srv._generate_structured_data("Answer.", _state(client))
        self.assertEqual(list(out.keys()), ["timeline"])
        prompt = client.calls[0]["messages"][1]["content"]
        self.assertNotIn("mindMap", prompt)


class ChatRequestFieldsTests(unittest.TestCase):
    def test_the_websocket_filter_keeps_the_map_fields(self):
        # Same filter the /chat-stream handler applies to incoming JSON.
        data = {
            "user_input": "Please generate a visual strategy map for this case.",
            "mind_map_requested": True,
            "case_mind_map_context": "Case: Cruz v. Reyes",
            "not_a_field": 1,
        }
        request = srv.ChatRequest(**{k: v for k, v in data.items() if k in srv.ChatRequest.model_fields})
        self.assertTrue(request.mind_map_requested)
        self.assertEqual(request.case_mind_map_context, "Case: Cruz v. Reyes")

    def test_both_fields_default_to_none(self):
        request = srv.ChatRequest(user_input="hi")
        self.assertIsNone(request.mind_map_requested)
        self.assertIsNone(request.case_mind_map_context)
        self.assertIsNone(request.skip_legal_verify)

    def test_the_websocket_filter_keeps_skip_legal_verify(self):
        data = {"user_input": "Build the case map.", "skip_legal_verify": True}
        request = srv.ChatRequest(**{k: v for k, v in data.items() if k in srv.ChatRequest.model_fields})
        self.assertTrue(request.skip_legal_verify)


if __name__ == "__main__":
    unittest.main()
