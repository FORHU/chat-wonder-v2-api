"""Tests for the hand-off of a drafted document to ilovelawyer-api (#69/#71).

The drafted text travels to ilovelawyer-api through the session state (last_generated_file_result,
sent as the [GENERATED_FILE_DATA] frame), not through the model. The model only gets a note: with
the full text in its context it pastes the whole document into the chat reply instead of pointing
at the downloadable file.
"""

import json
import unittest
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)


class DocumentHandoffTests(unittest.TestCase):
    def _run(self, tool_name: str, tool_result: dict):
        sid = "test-doc-handoff"
        srv._context.sessions[sid] = srv.ChatState()
        try:
            with patch.dict(srv.__dict__, {tool_name: lambda **kw: dict(tool_result)}):
                model_facing = srv.execute_function_call({"name": tool_name, "arguments": json.dumps({})}, session_id=sid)
            return model_facing, srv._context.sessions[sid].last_generated_file_result
        finally:
            srv._context.sessions.pop(sid, None)

    def test_document_is_captured_for_ilovelawyer_api_and_hidden_from_the_model(self):
        result = {
            "success": True,
            "content": "FULL DOCUMENT TEXT",
            "format": "pdf",
            "document_type": "affidavit_of_loss",
            "document_name": "Affidavit of Loss",
        }
        model_facing, captured = self._run("generate_legal_document", result)
        self.assertEqual(
            captured,
            {"content": "FULL DOCUMENT TEXT", "format": "pdf", "document_type": "affidavit_of_loss", "document_name": "Affidavit of Loss"},
        )
        self.assertNotIn("content", model_facing)
        self.assertNotIn("FULL DOCUMENT TEXT", json.dumps(model_facing))

    def test_model_is_told_to_link_the_document_inline_not_paste_it(self):
        result = {"success": True, "content": "TEXT", "format": "docx", "document_type": "demand_letter", "document_name": "Demand Letter"}
        model_facing, _captured = self._run("generate_legal_document", result)
        note = model_facing["document_delivery"]
        self.assertIn("#download", note)
        self.assertIn("Do NOT reproduce", note)
        # Other fields the model may want to mention are left intact.
        self.assertEqual(model_facing["document_name"], "Demand Letter")

    def test_pleading_name_falls_back_to_pleading_type(self):
        result = {"success": True, "content": "TEXT", "format": "docx", "pleading_type": "Motion to Dismiss"}
        _model_facing, captured = self._run("draft_pleading", result)
        self.assertEqual(captured["document_name"], "Motion to Dismiss")
        self.assertEqual(captured["document_type"], "pleading")

    def test_failed_draft_is_not_captured_and_is_returned_unchanged(self):
        result = {"success": False, "error": "Missing required fields"}
        model_facing, captured = self._run("generate_legal_document", result)
        self.assertEqual(captured, {})
        self.assertEqual(model_facing, result)


if __name__ == "__main__":
    unittest.main()
