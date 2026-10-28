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
        self.assertEqual(tool_names, {"case_law_search", "judgment_get_header", "legislation_search", "legislation_get_section", "get_case_document"})
        # Never the legal_uk whitelist's research-chain/drafting tools — intake only informs,
        # it never builds an applied argument or drafts anything.
        for tool in ("draft_pleading_uk", "generate_legal_document_uk", "get_legal_recommendation_uk", "case_law_grep_judgment"):
            self.assertNotIn(tool, tool_names)
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


if __name__ == "__main__":
    unittest.main()
