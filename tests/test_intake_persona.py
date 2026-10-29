"""Case Intake: ilovelawyer-client's consumer-side consultation persona.

Gathers facts for a solicitor to review, and may explain the user's general UK rights
with real, cited sources — never applies that law to assess THEIR case, predict an
outcome, or tell them what to do (that stays legal_uk-only). A small, read-only tool
whitelist (case law / legislation lookup, case document read) backs the citations; it
is NOT the legal_uk whitelist and intake never enters `legal_mode` (no verifier/refine
loop, no doctrine guards) — see _finalize_intake_citations in the_server.py. Wiring-only
tests, same shape as UkWiringTests in tests/test_legal_recommendation_uk.py: no OpenAI
call, no live data.
"""

import os
import unittest
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)


class IntakeWiringTests(unittest.TestCase):
    def test_tag_resolves_to_the_intake_persona_with_its_small_tool_whitelist(self):
        persona, cleaned, tools, addendum = srv.process_persona("[intake] my landlord won't fix the boiler")
        self.assertEqual(persona, "intake")
        tool_names = {t["function"]["name"] for t in tools}
        self.assertEqual(
            tool_names,
            {"case_law_search", "judgment_get_header", "legislation_search", "legislation_get_section", "find_lawyer", "mark_case_ready"},
        )
        # Never the legal_uk whitelist's research-chain/drafting tools — intake only informs,
        # it never builds an applied argument or drafts anything.
        for tool in ("draft_pleading_uk", "generate_legal_document_uk", "get_legal_recommendation_uk", "case_law_grep_judgment"):
            self.assertNotIn(tool, tool_names)
        # Not get_case_document either: its scope check depends on sync_active_case_documents,
        # which intake never calls, so whitelisting it would let any consumer fetch any case's
        # document by UUID with no check at all.
        self.assertNotIn("get_case_document", tool_names)
        self.assertEqual(cleaned, "my landlord won't fix the boiler")
        self.assertTrue(addendum)

    def test_the_prompt_forbids_applying_law_to_the_users_facts(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] do I have a case?")
        self.assertIn("NEVER", addendum)
        self.assertIn("conclude whether THIS user has a case", addendum)

    def test_the_prompt_allows_general_rights_info_with_citations(self):
        # The advice refinement (2026-10-28): intake may now explain general UK rights, with
        # real citations, as long as it never applies them to the user's specific facts.
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] hello")
        self.assertIn("RIGHTS & CITATIONS", addendum)
        self.assertIn("Never hand-construct a", addendum)
        self.assertIn("general information, not analysis of their facts", addendum)

    def test_the_prompt_describes_what_to_gather(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] hello")
        for field in ("part of the UK", "the story in the user's own words", "what the user wants to happen"):
            self.assertIn(field, addendum)

    def test_legal_and_legal_uk_personas_are_unaffected(self):
        legal_persona, _c, legal_tools, _a = srv.process_persona("[legal ai] what are my rights")
        self.assertEqual(legal_persona, "legal")
        self.assertTrue(legal_tools)  # the PH whitelist, untouched

        uk_persona, _c, uk_tools, _a = srv.process_persona("[legal ai uk] what are my rights")
        self.assertEqual(uk_persona, "legal_uk")
        self.assertTrue(uk_tools)  # the UK whitelist, untouched

    def test_intake_never_enters_legal_mode(self):
        # `_finalize_legal_response` runs for every persona (it isn't gated by persona id), but its
        # citation-gate, quote-strip and doctrine-guard steps are gated internally by `legal_mode`,
        # which checks for this literal string in the addendum (see _finalize_legal_response /
        # apply_legal_citation_pipeline). The intake prompt must never contain it.
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] my landlord won't fix the boiler")
        self.assertNotIn("LEGAL ASSISTANT MODE", addendum)

    def test_the_prompt_forbids_declaring_readiness_without_checking_each_item(self):
        # Regression for a live bug (2026-10-26): after two short exchanges the model declared
        # "I think I have the main information needed" and wrote "You want to seek money" into
        # the readiness summary — the user never said that. The prompt must explicitly forbid
        # summarizing an unasked/unanswered must-have item as settled, and call out the specific
        # item that got fabricated.
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] hello")
        self.assertIn("a guess is not an answer", addendum)
        self.assertIn("do not write a readiness summary", addendum)
        self.assertIn("do not include it in a\nreadiness summary as if it were settled", addendum)

    def test_the_prompt_requires_mark_case_ready_the_tool_call_not_prose(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] hello")
        self.assertIn("mark_case_ready()", addendum)
        self.assertIn("not enough", addendum)

    def test_the_prompt_tells_intake_to_find_a_lawyer_on_request_without_waiting_for_the_checklist(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] hello")
        self.assertIn("find_lawyer", addendum)
        self.assertIn("Never wait for the checklist to be\ncomplete", addendum)

    def test_the_prompt_checks_document_excerpts_too_before_asking_again(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] hello")
        self.assertIn("document excerpts", addendum)

    def test_finalize_intake_citations_strips_an_unverified_link_but_keeps_a_verified_one(self):
        # _finalize_intake_citations reuses legal_citations.py's pure gate functions directly
        # (not the legal/legal_uk persona gate) — a citation backed by this turn's tool results
        # survives, one the model invented does not, demoted to plain text rather than deleted.
        verified_url = "https://www.legislation.gov.uk/ukpga/1996/18/section/94"
        search_results = [{"url": verified_url}]
        text = (
            f"Under [Employment Rights Act 1996, s 94]({verified_url}) you generally have the "
            "right not to be unfairly dismissed, as discussed in "
            "[a case the AI made up](https://example.com/not-a-real-case)."
        )
        out = srv._finalize_intake_citations(text, search_results)
        self.assertIn(verified_url, out)
        self.assertNotIn("example.com/not-a-real-case", out)
        self.assertIn("a case the AI made up", out)

    def test_intake_reasoning_effort_falls_back_to_none_without_the_responses_api(self):
        # Regression for a live bug (2026-10-28): intake's new tool whitelist made the old
        # "always low effort, always Chat Completions" pairing invalid — gpt-5.6-luna rejects
        # function tools + a real reasoning_effort on /v1/chat/completions ("Function tools with
        # reasoning_effort are not supported... To use function tools, use /v1/responses or set
        # reasoning_effort to 'none'."). _intake_model_override must follow the same
        # Responses-API-or-none pattern _legal_model_override already uses, or reason_loop's
        # routing (persona == "intake" and _legal_use_responses_api()) would send "low" straight
        # into the Chat Completions path whenever that flag is off.
        with patch.dict(os.environ, {"LEGAL_USE_RESPONSES_API": "true", "INTAKE_REASONING_EFFORT": ""}):
            _model, effort = srv._intake_model_override("intake")
            self.assertEqual(effort, "low")
        with patch.dict(os.environ, {"LEGAL_USE_RESPONSES_API": "false", "INTAKE_REASONING_EFFORT": ""}):
            _model, effort = srv._intake_model_override("intake")
            self.assertEqual(effort, "none")

    def test_both_persona_label_dicts_know_about_intake(self):
        import re

        source = open("the_server.py", encoding="utf-8").read()
        label_dicts = re.findall(r'_persona_label = \{[^}]*\}', source)
        self.assertGreaterEqual(len(label_dicts), 2)
        for d in label_dicts:
            self.assertIn('"intake"', d)

    def test_a_new_session_starts_with_no_find_lawyer_or_ready_state(self):
        state = srv.ChatState()
        self.assertEqual(state.last_find_lawyer_result, [])
        self.assertFalse(state.last_mark_case_ready)

    def test_intake_turns_reset_find_lawyer_and_ready_state_from_a_prior_turn(self):
        # Regression guard for the staleness the stylist persona's last_outfit_ids_result has (no
        # reset, so it resends stale ids on every later turn once set) — intake must not inherit
        # that. process_persona itself has no session (it's a pure dispatcher); the reset is the
        # websocket handler's job, in the same elif as legal's last_generated_file_result reset.
        # Exercise that exact branch directly, the way the handler calls it.
        session_id = "test-intake-reset"
        state = srv.ChatState()
        state.last_find_lawyer_result = ["lawyer-1"]
        state.last_mark_case_ready = True
        srv._context.sessions[session_id] = state
        try:
            persona, _cleaned, _tools, _addendum = srv.process_persona("[intake] anything else I should know?")
            self.assertEqual(persona, "intake")
            if persona == "intake" and session_id in srv._context.sessions:
                srv._context.sessions[session_id].last_find_lawyer_result = []
                srv._context.sessions[session_id].last_mark_case_ready = False
            self.assertEqual(srv._context.sessions[session_id].last_find_lawyer_result, [])
            self.assertFalse(srv._context.sessions[session_id].last_mark_case_ready)
        finally:
            del srv._context.sessions[session_id]


class FindLawyerAndMarkCaseReadyTests(unittest.TestCase):
    """find_lawyer/mark_case_ready, loaded into srv's globals by _load_user_functions (see the
    module-level call at the top of this file). Each function's own __globals__ — not srv's, and
    not a freshly re-imported module — is what _http_post_json patching must target, since that's
    the namespace Python actually resolves the name against at call time."""

    def setUp(self):
        self.find_lawyer = srv.find_lawyer
        self.mark_case_ready = srv.mark_case_ready

    def _patch_http_post_json(self, fake):
        original = self.find_lawyer.__globals__["_http_post_json"]
        self.find_lawyer.__globals__["_http_post_json"] = fake
        self.addCleanup(lambda: self.find_lawyer.__globals__.__setitem__("_http_post_json", original))

    def test_mark_case_ready_always_succeeds_with_no_args(self):
        self.assertEqual(self.mark_case_ready(), {"success": True})

    def test_find_lawyer_refuses_no_case_areas(self):
        self.assertEqual(self.find_lawyer(case_areas_json="[]")["success"], False)
        self.assertEqual(self.find_lawyer(case_areas_json="not json")["success"], False)

    def test_find_lawyer_reports_missing_config_rather_than_crashing(self):
        with patch.dict(os.environ, {"ILOVELAWYER_API_BASE": "", "CHAT_WONDER_API_KEY": ""}):
            result = self.find_lawyer(case_areas_json='["EMPLOYMENT"]', uk_jurisdiction="England and Wales")
        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"], "missing_config")

    def test_find_lawyer_posts_criteria_and_caps_to_five_ranked_ids(self):
        captured = {}

        def fake_post(url, body, headers):
            captured["url"] = url
            captured["body"] = body
            captured["headers"] = headers
            return {"matches": [{"lawyerId": f"lawyer-{i}", "budgetFitScore": i} for i in range(8)]}

        self._patch_http_post_json(fake_post)
        with patch.dict(os.environ, {"ILOVELAWYER_API_BASE": "http://localhost:3001", "CHAT_WONDER_API_KEY": "test-key"}):
            result = self.find_lawyer(case_areas_json='["EMPLOYMENT"]', uk_jurisdiction="England and Wales", budget_min_minor=50_000, budget_max_minor=150_000)

        self.assertEqual(result["success"], True)
        self.assertEqual(result["ids"], ["lawyer-0", "lawyer-1", "lawyer-2", "lawyer-3", "lawyer-4"])
        self.assertEqual(captured["url"], "http://localhost:3001/api/v1/lawyer-matching/by-criteria")
        self.assertEqual(
            captured["body"],
            {"caseAreas": ["EMPLOYMENT"], "ukJurisdiction": "England and Wales", "budget": {"minMinor": 50_000, "maxMinor": 150_000, "feePreference": None}},
        )
        self.assertEqual(captured["headers"], {"x-api-key": "test-key"})

    def test_find_lawyer_omits_budget_entirely_when_none_given(self):
        captured = {}
        self._patch_http_post_json(lambda url, body, headers: captured.setdefault("body", body) and {"matches": []} or {"matches": []})
        with patch.dict(os.environ, {"ILOVELAWYER_API_BASE": "http://localhost:3001", "CHAT_WONDER_API_KEY": "test-key"}):
            result = self.find_lawyer(case_areas_json='["EMPLOYMENT"]')

        self.assertEqual(result, {"success": True, "ids": [], "message": "No currently available lawyers match this case yet."})
        self.assertIsNone(captured["body"]["budget"])

    def test_find_lawyer_reports_the_upstream_http_error_rather_than_crashing(self):
        import urllib.error

        self._patch_http_post_json(lambda url, body, headers: (_ for _ in ()).throw(urllib.error.HTTPError(url, 500, "boom", {}, None)))
        with patch.dict(os.environ, {"ILOVELAWYER_API_BASE": "http://localhost:3001", "CHAT_WONDER_API_KEY": "test-key"}):
            result = self.find_lawyer(case_areas_json='["EMPLOYMENT"]')

        self.assertEqual(result["success"], False)
        self.assertEqual(result["error"], "http_500")


if __name__ == "__main__":
    unittest.main()
