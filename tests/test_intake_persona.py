"""Case Intake: ilovelawyer-client's consumer-side consultation persona.

Gathers facts for a solicitor to review; never gives legal advice. No tools — nothing
to look up, everything comes from the user. Wiring-only tests, same shape as
UkWiringTests in tests/test_legal_recommendation_uk.py: no OpenAI call, no live data.
"""

import unittest

import the_server as srv

srv._load_user_functions(overwrite_globals=True)


class IntakeWiringTests(unittest.TestCase):
    def test_tag_resolves_to_the_intake_persona_with_no_tools(self):
        persona, cleaned, tools, addendum = srv.process_persona("[intake] my landlord won't fix the boiler")
        self.assertEqual(persona, "intake")
        self.assertEqual(tools, [])
        self.assertEqual(cleaned, "my landlord won't fix the boiler")
        self.assertTrue(addendum)

    def test_the_prompt_forbids_applying_law_to_the_users_facts(self):
        _persona, _cleaned, _tools, addendum = srv.process_persona("[intake] do I have a case?")
        self.assertIn("NEVER", addendum)
        self.assertIn("assess whether they have a case", addendum)

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

    def test_both_persona_label_dicts_know_about_intake(self):
        import re

        source = open("the_server.py", encoding="utf-8").read()
        label_dicts = re.findall(r'_persona_label = \{[^}]*\}', source)
        self.assertGreaterEqual(len(label_dicts), 2)
        for d in label_dicts:
            self.assertIn('"intake"', d)


if __name__ == "__main__":
    unittest.main()
