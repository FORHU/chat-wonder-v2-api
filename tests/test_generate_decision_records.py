"""End-to-end test of the_server._generate_decision_records against a scripted fake OpenAI
client — the plumbing (prompt assembly, JSON extraction, audit, metrics) around the pure
legal_decisions.audit_decision_records logic already covered in tests/test_legal_decisions.py.
No live OpenAI call.
"""

import json
import unittest
from types import SimpleNamespace

import the_server as srv

srv._load_user_functions(overwrite_globals=True)

POOL = [{"id": "abc", "title": "Manalo v. Republic", "url": "https://juris.ph/case/abc"}]
MANIFEST = [{"id": "doc-1", "name": "D01_HSE_Preliminary_Investigation_Report.pdf"}]
CASE_DOCS = [{"id": "doc-1", "name": "D01_HSE_Preliminary_Investigation_Report.pdf", "text": "The east tie line at levels 6 and 7 was found incomplete."}]
ANSWER = "The absence of the through-ties was a substantial cause of the collapse."

GOOD_RECORD = {
    "records": [
        {
            "anchor": ANSWER,
            "conclusion": "The missing ties caused the collapse.",
            "rule": [{"title": "CMCHA 2007, s 1", "url": "https://juris.ph/case/abc"}],
            "evidenceFor": [{"doc": "D01", "pinpoint": "para 10", "quote": "east tie line at levels 6 and 7 was found incomplete"}],
            "evidenceAgainst": [],
            "alternatives": [],
            "weighting": "The survey is uncontested.",
            "confidence": "high",
            "wouldChangeIf": ["The ties were found intact"],
        }
    ]
}


def _completion_returning(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _completion_returning(self.content)


def _state(client, case_documents=None, manifest=None):
    state = srv.ChatState()
    state.openai_client = client
    state.last_search_legal_results = list(POOL)
    state.active_case_documents = case_documents or []
    state.case_document_manifest = manifest or []
    return state


class GenerateDecisionRecordsTests(unittest.TestCase):
    def test_returns_verified_records_from_a_clean_model_response(self):
        client = _FakeClient(json.dumps(GOOD_RECORD))
        state = _state(client, CASE_DOCS, MANIFEST)
        out = srv._generate_decision_records("Is Meridian liable?", ANSWER, POOL, [], state)
        self.assertIsNotNone(out)
        self.assertEqual(len(out["records"]), 1)
        rec = out["records"][0]
        self.assertTrue(rec["rule"][0]["verified"])
        self.assertTrue(rec["evidenceFor"][0]["verified"])
        self.assertEqual(rec["evidenceFor"][0]["docId"], "doc-1")
        # prompt actually includes the manifest and the answer text
        sent_prompt = client.calls[0]["messages"][1]["content"]
        self.assertIn("D01", sent_prompt)
        self.assertIn(ANSWER, sent_prompt)

    def test_returns_none_on_empty_response_text(self):
        state = _state(_FakeClient("{}"))
        self.assertIsNone(srv._generate_decision_records("q", "", POOL, [], state))

    def test_returns_none_when_all_records_fail_anchor_check(self):
        bad = {"records": [{**GOOD_RECORD["records"][0], "anchor": "not present anywhere in the answer"}]}
        client = _FakeClient(json.dumps(bad))
        state = _state(client, CASE_DOCS, MANIFEST)
        self.assertIsNone(srv._generate_decision_records("q", ANSWER, POOL, [], state))

    def test_tolerates_a_json_code_fence_around_the_response(self):
        fenced = "```json\n" + json.dumps(GOOD_RECORD) + "\n```"
        client = _FakeClient(fenced)
        state = _state(client, CASE_DOCS, MANIFEST)
        out = srv._generate_decision_records("q", ANSWER, POOL, [], state)
        self.assertIsNotNone(out)
        self.assertEqual(len(out["records"]), 1)

    def test_malformed_json_from_model_fails_closed(self):
        client = _FakeClient("not json at all")
        state = _state(client, CASE_DOCS, MANIFEST)
        self.assertIsNone(srv._generate_decision_records("q", ANSWER, POOL, [], state))

    def test_openai_exception_fails_closed(self):
        class RaisingClient:
            chat = SimpleNamespace(completions=SimpleNamespace(create=lambda **kw: (_ for _ in ()).throw(RuntimeError("boom"))))

        state = _state(RaisingClient())
        self.assertIsNone(srv._generate_decision_records("q", ANSWER, POOL, [], state))


if __name__ == "__main__":
    unittest.main()
