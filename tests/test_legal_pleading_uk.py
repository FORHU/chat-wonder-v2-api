"""Tests for the England & Wales pleading tool (draft_pleading_uk),
their persona wiring, and the hand-off of drafted text to ilovelawyer-api for rendering. No live
OpenAI call — the client is faked the same way tests/test_draft_pleading.py fakes it.
"""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)
import resources.functions.user_functions as uf  # noqa: E402  (loaded above)


class _FakeClient:
    def __init__(self, content: str = "DRAFTED UK TEXT"):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.content))])


def _user_prompt(fake: _FakeClient) -> str:
    return next(m["content"] for m in fake.calls[0]["messages"] if m["role"] == "user")


class DraftPleadingUkTests(unittest.TestCase):
    def test_missing_required_fields_returns_error_without_calling_openai(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            r = uf.draft_pleading_uk(pleading_type="", case_facts="", grounds="")
        self.assertFalse(r["success"])
        self.assertEqual(set(r["missing_fields"]), {"pleading_type", "case_facts", "grounds"})
        self.assertEqual(fake.calls, [])

    def test_facts_grounds_and_cpr_rules_reach_the_prompt(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            r = uf.draft_pleading_uk(
                pleading_type="Defence",
                case_facts="UNIQUE_FACT_MARKER_998",
                grounds="UNIQUE_GROUNDS_MARKER_777",
                court="County Court at Leeds",
                claim_number="K0LS1234",
                party_role="Defendant",
            )
        self.assertTrue(r["success"])
        self.assertEqual(r["content"], "DRAFTED UK TEXT")
        self.assertEqual(r["pleading_type"], "Defence")
        prompt = _user_prompt(fake)
        for expected in ("UNIQUE_FACT_MARKER_998", "UNIQUE_GROUNDS_MARKER_777", "County Court at Leeds", "K0LS1234", "CPR 16.5(1)", "CPR 16.4(1)(a)", "contempt of court"):
            self.assertIn(expected, prompt)

    def test_prompt_forbids_inventing_interest_statutes_or_extra_heads_of_claim(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading_uk(pleading_type="Particulars of Claim", case_facts="Unpaid £4,200.", grounds="Breach of contract.")
        prompt = _user_prompt(fake)
        self.assertIn("never state a rate or a statutory basis the user did not give", prompt)
        self.assertIn("Do not cite case law or any statute", prompt)
        self.assertIn("Do not add alternative causes of action", prompt)

    def test_blanks_use_uk_style_placeholder(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading_uk(pleading_type="Reply", case_facts="Some facts.", grounds="Some grounds.")
        self.assertIn("Court: [___]", _user_prompt(fake))

    def test_authorities_lift_the_case_law_ban(self):
        """#73: when verified authorities are supplied, the outright case-law/statute ban
        must be replaced with an instruction to cite only from that list."""
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading_uk(
                pleading_type="Particulars of Claim",
                case_facts="Unpaid invoice.",
                grounds="Breach of contract.",
                authorities=[
                    {"citation": "De Bank Haycocks v ADP RPO UK Ltd [2024] EWCA Civ 1291", "proposition": "unfair dismissal on consultation grounds"},
                ],
            )
        prompt = _user_prompt(fake)
        self.assertIn("De Bank Haycocks v ADP RPO UK Ltd [2024] EWCA Civ 1291", prompt)
        self.assertIn("unfair dismissal on consultation grounds", prompt)
        self.assertNotIn("Do not cite case law or any statute.", prompt)
        self.assertIn("Do not cite any case law or statute that is not listed under Authorities", prompt)

    def test_no_authorities_keeps_the_existing_cpr_only_ban(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading_uk(pleading_type="Defence", case_facts="Facts.", grounds="Grounds.")
        prompt = _user_prompt(fake)
        self.assertIn("Cite only the CPR rules and practice directions named above. Do not cite case law or any statute.", prompt)

    def test_manifest_declares_authorities_field_on_draft_pleading_uk(self):
        tool = next(x["function"] for x in srv._context.all_fun_manifest if x["function"]["name"] == "draft_pleading_uk")
        self.assertIn("authorities", tool["parameters"]["properties"])
        self.assertNotIn("authorities", tool["parameters"]["required"])


class UkPleadingWiringTests(unittest.TestCase):
    @staticmethod
    def _tool_names(tag: str) -> set:
        _persona, _cleaned, tools, _addendum = srv.process_persona(tag)
        return {t["function"]["name"] for t in (tools or [])}

    def test_uk_persona_has_draft_pleading_uk_and_not_the_philippine_pleading_tool(self):
        names = self._tool_names("[legal ai uk] draft particulars of claim")
        self.assertIn("draft_pleading_uk", names)
        self.assertNotIn("draft_pleading", names)

    def test_philippine_persona_does_not_get_draft_pleading_uk(self):
        self.assertNotIn("draft_pleading_uk", self._tool_names("[legal ai] draft a motion"))

    def test_uk_prompt_offers_pleading_drafting(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[legal ai uk] draft a defence")
        self.assertIn("draft_pleading_uk", addendum)

    def test_manifest_declares_draft_pleading_uk_with_its_required_fields(self):
        tool = next(x["function"] for x in srv._context.all_fun_manifest if x["function"]["name"] == "draft_pleading_uk")
        self.assertEqual(set(tool["parameters"]["required"]), {"pleading_type", "case_facts", "grounds"})
        self.assertEqual(tool["parameters"]["properties"]["format"]["enum"], ["docx", "pdf"])


class UkPleadingHandoffTests(unittest.TestCase):
    """The drafted text goes to ilovelawyer-api through the session state, not through the model."""



    def _run(self, tool_name: str, persona_result: dict):
        sid = "test-doc-handoff"
        srv._context.sessions[sid] = srv.ChatState()
        try:
            with patch.dict(srv.__dict__, {tool_name: lambda **kw: dict(persona_result)}):
                model_facing = srv.execute_function_call({"name": tool_name, "arguments": json.dumps({})}, session_id=sid)
            return model_facing, srv._context.sessions[sid].last_generated_file_result
        finally:
            srv._context.sessions.pop(sid, None)

    def test_uk_pleading_is_captured_using_its_pleading_type_as_the_name(self):
        result = {"success": True, "content": "FULL PLEADING", "format": "docx", "pleading_type": "Defence"}
        model_facing, captured = self._run("draft_pleading_uk", result)
        self.assertEqual(captured["document_name"], "Defence")
        self.assertEqual(captured["document_type"], "pleading")
        self.assertNotIn("content", model_facing)


if __name__ == "__main__":
    unittest.main()
