"""Phase 0 of the differentiation program — the Brackenmoor benchmark fixes.

Covers: partial-document flag on get_case_document and the cache/manifest that consumes it
(0.1), whole-case preload via case_document_texts (0.2), tool budget scaled by sub-question
count (0.5), UK doctrine guards routed by jurisdiction (0.6), and bundle-refs split out of a
citation link label (0.7). No live OpenAI / MCP / ilovelawyer-api calls.
"""

import asyncio
import unittest
from unittest.mock import patch

import the_server as srv
from legal_citations import format_legal_citation_links, split_bundle_refs_out_of_labels
from legal_fact_boost import append_critical_doctrine_guards, append_uk_doctrine_guards

srv._load_user_functions(overwrite_globals=True)
import resources.functions.user_functions as uf  # noqa: E402  (loaded above)


def _fake_api(chunks):
    return {"name": "D01.pdf", "ragStatus": "READY", "chunks": chunks}


CHUNKS = [{"id": f"c{i}", "chunkIndex": i, "chunkText": f"para {i}"} for i in range(6)]


class GetCaseDocumentPartialTests(unittest.TestCase):
    def test_full_read_is_not_partial(self):
        with patch.object(uf, "_http_get_json", return_value=_fake_api(CHUNKS)), \
             patch.dict("os.environ", {"ILOVELAWYER_API_BASE": "http://x", "CHAT_WONDER_API_KEY": "k"}):
            r = uf.get_case_document("d1")
        self.assertTrue(r["success"])
        self.assertFalse(r["partial"])
        self.assertEqual(r["document_chunk_count"], 6)
        self.assertNotIn("note", r)

    def test_chunk_filtered_read_is_partial_with_note(self):
        with patch.object(uf, "_http_get_json", return_value=_fake_api(CHUNKS)), \
             patch.dict("os.environ", {"ILOVELAWYER_API_BASE": "http://x", "CHAT_WONDER_API_KEY": "k"}):
            r = uf.get_case_document("d1", case_document_chunk_ids=["c1", "c4"])
        self.assertTrue(r["partial"])
        self.assertEqual(r["chunk_count"], 2)
        self.assertEqual(r["document_chunk_count"], 6)
        self.assertIn("PARTIAL: 2 of 6", r["note"])
        self.assertIn("WITHOUT case_document_chunk_ids", r["note"])

    def test_paged_read_is_partial_until_last_page(self):
        with patch.object(uf, "_http_get_json", return_value=_fake_api(CHUNKS)), \
             patch.dict("os.environ", {"ILOVELAWYER_API_BASE": "http://x", "CHAT_WONDER_API_KEY": "k"}):
            first = uf.get_case_document("d1", offset=0, limit=4)
            last = uf.get_case_document("d1", offset=4, limit=4)
        self.assertTrue(first["partial"])
        self.assertIn("offset=4", first["note"])
        self.assertFalse(last["partial"])


class CacheAndManifestTests(unittest.TestCase):
    def _session(self):
        sid = "phase0-cache"
        state = srv.ChatState()
        state.allowed_case_document_ids = {"d1"}
        state.case_document_manifest = [{"id": "d1", "name": "D01.pdf", "category": "HSE report"}]
        srv._context.sessions[sid] = state
        return sid, state

    def _fetch(self, sid, result):
        with patch.dict(srv.__dict__, {"get_case_document": lambda **kw: result}):
            srv.execute_function_call({"name": "get_case_document", "arguments": '{"case_document_id": "d1"}'}, session_id=sid)

    def test_partial_read_never_replaces_full_cached_document(self):
        sid, state = self._session()
        try:
            self._fetch(sid, {"success": True, "id": "d1", "name": "D01.pdf", "text": "full text of D01", "chunk_count": 6, "document_chunk_count": 6, "partial": False})
            self._fetch(sid, {"success": True, "id": "d1", "name": "D01.pdf", "text": "para 1", "chunk_count": 1, "document_chunk_count": 6, "partial": True})
            self.assertEqual(state.case_document_cache["d1"]["text"], "full text of D01")
            self.assertFalse(state.case_document_cache["d1"]["partial"])
        finally:
            srv._context.sessions.pop(sid, None)

    def test_manifest_marks_partial_documents_and_how_to_fix(self):
        sid, state = self._session()
        try:
            self._fetch(sid, {"success": True, "id": "d1", "name": "D01.pdf", "text": "para 1", "chunk_count": 1, "document_chunk_count": 6, "partial": True})
            block = srv._build_case_document_injection(state)
            self.assertIn("PARTIAL in context (1 of 6 chunks", block)
            self.assertIn("NO case_document_chunk_ids", block)
            self._fetch(sid, {"success": True, "id": "d1", "name": "D01.pdf", "text": "full text", "chunk_count": 6, "document_chunk_count": 6, "partial": False})
            self.assertIn("in context in full", srv._build_case_document_injection(state))
        finally:
            srv._context.sessions.pop(sid, None)

    def test_case_document_texts_preload_seeds_full_entries(self):
        sid = "phase0-preload"
        srv._context.sessions[sid] = srv.ChatState()
        try:
            asyncio.run(srv.sync_active_case_documents(
                sid, ["d1", "d2"], None,
                [{"id": "d1", "name": "D01.pdf", "category": None}, {"id": "d2", "name": "D02.pdf", "category": None}],
                [{"id": "d1", "name": "D01.pdf", "text": "whole of D01"}, {"id": "d2", "name": "D02.pdf", "text": "whole of D02"}, {"id": "other", "name": "x", "text": "not allowed"}],
            ))
            state = srv._context.sessions[sid]
            self.assertEqual({d["id"] for d in state.active_case_documents}, {"d1", "d2"})
            self.assertTrue(all(d["preloaded"] and not d["partial"] for d in state.active_case_documents))
            self.assertNotIn("other", state.case_document_cache)
            self.assertIn("in context in full", srv._build_case_document_injection(state))
        finally:
            srv._context.sessions.pop(sid, None)


class SubQuestionBudgetTests(unittest.TestCase):
    def test_counts_numbered_and_lettered_parts(self):
        self.assertEqual(srv._count_sub_questions("Advise on:\n\n1.1 First\n\n1.2 Second\n\n1.3 Third"), 3)
        self.assertEqual(srv._count_sub_questions("(a) one\n(b) two"), 2)
        self.assertEqual(srv._count_sub_questions("What is the compensatory award cap?"), 1)
        self.assertEqual(srv._count_sub_questions("Quorum rules for a board"), 1)

    def test_budget_grows_with_parts_and_respects_ceiling(self):
        state = srv.ChatState()
        state.case_document_manifest = [{"id": str(i)} for i in range(20)]
        with patch.dict("os.environ", {"LEGAL_MAX_CHAINS": "20", "LEGAL_CHAINS_PER_SUBQUESTION": "6", "LEGAL_MAX_CHAINS_CEILING": "80"}):
            _, _, _, one = srv._legal_model_override("legal_uk", state, "single question")
            _, _, _, four = srv._legal_model_override("legal_uk", state, "1.1 a\n2.1 b\n2.2 c\n2.3 d")
        self.assertEqual(one, 60)
        self.assertEqual(four, 78)


class UkDoctrineGuardTests(unittest.TestCase):
    def test_whistleblowing_burden_and_remedies_added_when_silent(self):
        q = "She has less than two years' service and claims automatically unfair dismissal under s.103A."
        out = append_uk_doctrine_guards("The s.103A claim is strong.", q)
        self.assertIn("Burden without qualifying service", out)
        self.assertIn("124(1A)", out)

    def test_whistleblowing_guard_silent_when_answer_covers_it(self):
        q = "She has less than two years' service and claims under s.103A."
        text = "The burden lies on the claimant. The award is uncapped under s.124(1A); an ACAS uplift may apply."
        self.assertEqual(append_uk_doctrine_guards(text, q), text)

    def test_dpa_part3_trap(self):
        q = "Covert monitoring — lawful under Data Protection Act 2018 ss 35-40?"
        out = append_uk_doctrine_guards("The employer must consider necessity.", q)
        self.assertIn("Part 3 does not apply to a private employer", out)

    def test_business_records_route(self):
        q = "Hearsay problems with the Met Office certificate and the Halbrook ledger."
        out = append_uk_doctrine_guards("These records may be admissible.", q)
        self.assertIn("CJA 2003 s.117", out)
        self.assertEqual(append_uk_doctrine_guards("Admissible under s 117 CJA 2003.", q), "Admissible under s 117 CJA 2003.")

    def test_northern_ireland_extent(self):
        q = "I work in Belfast. Does the Employment Rights Act 1996 protect me as a whistleblower?"
        out = append_uk_doctrine_guards("ERA 1996 s.43B protects disclosures.", q)
        self.assertIn("Northern Ireland has its own mirror legislation", out)

    def test_ph_guards_do_not_fire_on_uk_defamation_via_finalizer(self):
        text = "Defamation claim under the Defamation Act 2013 — serious harm must be shown."
        out = srv._finalize_legal_response(text, [], legal_mode=True, user_input="online libel about my business in Leeds", jurisdiction="UK")
        self.assertNotIn("Article 33", out)
        ph = srv._finalize_legal_response(text, [], legal_mode=True, user_input="online libel about my business in Leeds", jurisdiction="PH")
        self.assertIn("Article 33", ph)


class LinkLabelTests(unittest.TestCase):
    def test_bundle_refs_split_out_of_authority_label(self):
        t = "[D01, paras 3-5; D06, paras 2-4; Health and Safety at Work etc. Act 1974, s 3 Law](https://www.legislation.gov.uk/ukpga/1974/37/section/3)"
        out = split_bundle_refs_out_of_labels(t)
        self.assertEqual(out, "[D01, paras 3-5; D06, paras 2-4;] [Health and Safety at Work etc. Act 1974, s 3 Law](https://www.legislation.gov.uk/ukpga/1974/37/section/3)")
        html = format_legal_citation_links(t)
        self.assertIn(">Health and Safety at Work etc. Act 1974, s 3 Law</a>", html)
        self.assertNotIn(">D01", html)

    def test_plain_label_untouched(self):
        t = "[Employment Rights Act 1996, s 94 Law](https://www.legislation.gov.uk/ukpga/1996/18/section/94)"
        self.assertEqual(split_bundle_refs_out_of_labels(t), t)


if __name__ == "__main__":
    unittest.main()
