"""Tests for the England & Wales document tool (generate_legal_document_uk),
their persona wiring, and the hand-off of drafted text to ilovelawyer-api for rendering. No live
OpenAI call — the client is faked the same way the other tool tests fake it.
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


SAMPLE_DETAILS = {
    "witness_statement": dict(witness_name="Ann Smith", witness_address="1 High St, Leeds", party_role="Claimant", statement_facts="I paid the deposit on 1 May 2026."),
    "statutory_declaration": dict(declarant_name="Ann Smith", declarant_address="1 High St, Leeds", declaration_facts="I lost my passport on 3 June 2026."),
    "letter_before_claim": dict(sender_name="Ann Smith", recipient_name="Bob Jones", claim_summary="Unpaid invoice", remedy_sought="Pay £500"),
    "general_power_of_attorney": dict(donor_name="Ann Smith", attorney_name="Bob Jones"),
    "letter_of_authority": dict(authoriser_name="Ann Smith", authorised_person="Bob Jones", authority_purpose="Discuss my account"),
    "loan_agreement": dict(lender_name="Ann Smith", borrower_name="Bob Jones", principal_amount="£1,000", repayment_terms="£100 a month"),
    "deed_poll": dict(former_name="Ann Smith", new_name="Ann Jones", address="1 High St, Leeds"),
}


class GenerateLegalDocumentUkTests(unittest.TestCase):
    def test_every_template_has_sample_details_and_formats_cleanly(self):
        # Guards against a template referencing a {placeholder} that is neither a required nor an
        # optional field (KeyError at draft time) — and against a template with no sample here.
        self.assertEqual(set(SAMPLE_DETAILS), set(uf.UK_DOCUMENT_TEMPLATES))
        for doc_type, details in SAMPLE_DETAILS.items():
            with self.subTest(doc_type=doc_type):
                fake = _FakeClient()
                with patch("openai.OpenAI", return_value=fake):
                    r = uf.generate_legal_document_uk(document_type=doc_type, **details)
                self.assertTrue(r["success"], r)
                self.assertEqual(r["content"], "DRAFTED UK TEXT")
                self.assertEqual(r["format"], "docx")
                self.assertIn("England and Wales", fake.calls[0]["messages"][0]["content"])
                self.assertIn(list(details.values())[0], _user_prompt(fake))
                self.assertTrue(r["next_steps"])

    def test_missing_required_fields_returns_error_without_calling_openai(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            r = uf.generate_legal_document_uk(document_type="statutory_declaration", declarant_name="Ann Smith")
        self.assertFalse(r["success"])
        self.assertEqual(set(r["missing_fields"]), {"declarant_address", "declaration_facts"})
        self.assertEqual(fake.calls, [])

    def test_unknown_type_lists_available_types(self):
        r = uf.generate_legal_document_uk(document_type="affidavit_of_loss")  # Philippine-only type
        self.assertFalse(r["success"])
        self.assertEqual(set(r["available_types"]), set(uf.UK_DOCUMENT_TEMPLATES))

    def test_optional_fields_default_to_blanks(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            uf.generate_legal_document_uk(document_type="deed_poll", **SAMPLE_DETAILS["deed_poll"])
        self.assertIn("Date of birth: [___]", _user_prompt(fake))

    def test_statutory_declaration_prompt_carries_the_1835_wording(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            uf.generate_legal_document_uk(document_type="statutory_declaration", **SAMPLE_DETAILS["statutory_declaration"])
        prompt = _user_prompt(fake)
        self.assertIn("do solemnly and sincerely declare", prompt)
        self.assertIn("conscientiously believing the same to be true", prompt)
        self.assertIn("Statutory Declarations Act 1835", prompt)

    def test_witness_statement_prompt_carries_the_pd32_statement_of_truth(self):
        fake = _FakeClient()
        with patch("openai.OpenAI", return_value=fake):
            uf.generate_legal_document_uk(document_type="witness_statement", **SAMPLE_DETAILS["witness_statement"])
        self.assertIn("I believe that the facts stated in this witness statement are true.", _user_prompt(fake))

    def test_pdf_format_is_passed_through(self):
        with patch("openai.OpenAI", return_value=_FakeClient()):
            r = uf.generate_legal_document_uk(document_type="deed_poll", format="pdf", **SAMPLE_DETAILS["deed_poll"])
        self.assertEqual(r["format"], "pdf")


class UkManifestMatchesTemplatesTests(unittest.TestCase):
    """The model only sees the manifest. Every template field must be a declared parameter, labelled
    required/optional to match the template — otherwise the model asks for optional details (it did:
    it stalled a statutory declaration asking for the declarant's occupation) or never supplies a
    field the template needs."""

    def test_every_template_field_is_a_correctly_labelled_manifest_parameter(self):
        tool = next(
            t["function"] for t in srv._context.all_fun_manifest if t["function"]["name"] == "generate_legal_document_uk"
        )
        props = tool["parameters"]["properties"]
        self.assertEqual(set(tool["parameters"]["properties"]["document_type"]["enum"]), set(uf.UK_DOCUMENT_TEMPLATES))
        for doc_type, template in uf.UK_DOCUMENT_TEMPLATES.items():
            for field in template["required_fields"]:
                self.assertIn(field, props, f"{doc_type}.{field}")
                self.assertIn("(required)", props[field]["description"], f"{doc_type}.{field}")
            for field in template["optional_fields"]:
                self.assertIn(field, props, f"{doc_type}.{field}")
                self.assertNotIn("(required)", props[field]["description"], f"{doc_type}.{field}")
                self.assertIn("(optional)", props[field]["description"], f"{doc_type}.{field}")


class UkPersonaWiringTests(unittest.TestCase):
    @staticmethod
    def _tool_names(tag: str) -> set:
        _persona, _cleaned, tools, _addendum = srv.process_persona(tag)
        return {t["function"]["name"] for t in (tools or [])}

    def test_uk_persona_has_the_uk_document_tools_and_not_the_philippine_ones(self):
        names = self._tool_names("[legal ai uk] draft a statutory declaration")
        self.assertIn("generate_legal_document_uk", names)
        self.assertFalse({"generate_legal_document", "draft_pleading"} & names)

    def test_philippine_persona_does_not_get_the_uk_document_tools(self):
        names = self._tool_names("[legal ai] draft an affidavit")
        self.assertIn("generate_legal_document", names)
        self.assertNotIn("generate_legal_document_uk", names)

    def test_uk_prompt_no_longer_says_document_tools_are_unavailable(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[legal ai uk] draft a deed poll")
        self.assertIn("generate_legal_document_uk", addendum)
        self.assertNotIn("Those tools are not available in this legal persona whitelist. If asked, explain you can help with UK case law, legislation, and parliamentary research in this mode,", addendum)

    def test_no_prompt_still_tells_the_model_to_reproduce_the_document(self):
        for tag in ("[legal ai] draft an affidavit", "[legal ai uk] draft a deed poll"):
            _persona, _cleaned, _tools, addendum = srv.process_persona(tag)
            self.assertNotIn("reproduce its `content`", addendum, tag)


class DocumentHandoffTests(unittest.TestCase):
    """The drafted text goes to ilovelawyer-api via the session state, not via the model — the
    model only gets a note, otherwise it pastes the whole document into the chat reply."""

    def _run(self, tool_name: str, persona_result: dict):
        sid = "test-doc-handoff"
        srv._context.sessions[sid] = srv.ChatState()
        try:
            with patch.dict(srv.__dict__, {tool_name: lambda **kw: dict(persona_result)}):
                model_facing = srv.execute_function_call({"name": tool_name, "arguments": json.dumps({})}, session_id=sid)
            return model_facing, srv._context.sessions[sid].last_generated_file_result
        finally:
            srv._context.sessions.pop(sid, None)

    def test_uk_document_is_captured_and_hidden_from_the_model(self):
        result = {"success": True, "content": "FULL UK DOCUMENT", "format": "pdf", "document_type": "deed_poll", "document_name": "Deed Poll (Change of Name)"}
        model_facing, captured = self._run("generate_legal_document_uk", result)
        self.assertEqual(captured, {"content": "FULL UK DOCUMENT", "format": "pdf", "document_type": "deed_poll", "document_name": "Deed Poll (Change of Name)"})
        self.assertNotIn("content", model_facing)
        self.assertNotIn("FULL UK DOCUMENT", json.dumps(model_facing))
        self.assertIn("#download", model_facing["document_delivery"])

    def test_failed_draft_is_not_captured_and_is_returned_unchanged(self):
        result = {"success": False, "error": "Missing required fields"}
        model_facing, captured = self._run("generate_legal_document_uk", result)
        self.assertEqual(captured, {})
        self.assertEqual(model_facing, result)


if __name__ == "__main__":
    unittest.main()
