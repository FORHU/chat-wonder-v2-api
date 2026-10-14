"""Invariants for the UK post-generation doctrine guards (legal_fact_boost.UK_GUARDS).

A guard appends fixed text to a finished answer when keywords appear. The recurring bug was a
guard that fired on loose words in the wrong forum (a Crown Court s.117 note on an Employment
Tribunal answer) or asserted a fact about the user's case the question never stated. The contract:

  * the forum is read from the question and the case's own exhibits, never from the draft answer;
  * what a guard emits is true for any matter that can trigger it: forum-specific only when the
    matter positively matched that forum, otherwise a statement that holds in every forum;
  * every registered guard is exercised against every forum below, so a new guard added to
    UK_GUARDS is covered without writing a new test.
"""

import unittest

from legal_fact_boost import (
    UK_GUARDS,
    append_uk_doctrine_guards,
    case_forum_text,
    classify_uk_matter,
)

TRIGGER_STUFFED_ANSWER = (
    "The rota register, the invoice and the certificate are hearsay-adjacent. "
    "Unlike criminal proceedings the weight is for the decision-maker. "
    "DPA 2018 Part 3 ss 35-40 and a s.103A protected disclosure with less than two years' service "
    "are mentioned for completeness. Northern Ireland and Belfast are mentioned; ERA 1996 applies."
)

FORUM_QUESTIONS = {
    "criminal": "In the Crown Court prosecution, how is the hearsay in the Met Office certificate admitted?",
    "employment_tribunal": "How should the schedules of loss deal with double recovery in the Employment Tribunal?",
    "civil": "In the county court claim, how is the hearsay in the supplier's invoice treated?",
    "arbitration": "In the ad hoc arbitration, how should the tribunal treat the hearsay in the test register?",
    "unknown": "How is the hearsay in the Halbrook ledger treated?",
}


class ClassifierTests(unittest.TestCase):
    def test_forum_comes_from_the_question(self):
        for forum, question in FORUM_QUESTIONS.items():
            expected = set() if forum == "unknown" else {forum}
            self.assertEqual(set(classify_uk_matter(question).forums), expected, forum)

    def test_forum_can_come_from_the_case_exhibits_alone(self):
        class State:
            case_document_manifest = [{"name": "DEF-PLEAD-001_Grounds-of-Resistance.pdf", "category": ""}]
            case_document_cache = {"1": {"text": "IN THE EMPLOYMENT TRIBUNAL\nSITTING AT READING\nCase No. 3201847/2024"}}

        text = case_forum_text(State())
        self.assertEqual(set(classify_uk_matter("How should the schedules deal with it?", text).forums), {"employment_tribunal"})

    def test_a_bundle_can_span_several_forums(self):
        matter = classify_uk_matter("The Crown Court prosecution runs alongside an Employment Tribunal claim.")
        self.assertEqual(set(matter.forums), {"criminal", "employment_tribunal"})

    def test_case_forum_text_never_raises_without_a_case(self):
        self.assertEqual(case_forum_text(None), "")
        self.assertEqual(case_forum_text(object()), "")


class HearsayRecordsGuardTests(unittest.TestCase):
    def _out(self, forum, answer=TRIGGER_STUFFED_ANSWER):
        return append_uk_doctrine_guards(answer, FORUM_QUESTIONS[forum])

    def test_each_forum_gets_its_own_rule(self):
        self.assertIn("Business documents (criminal proceedings)", self._out("criminal"))
        self.assertIn("Hearsay and records (Employment Tribunal)", self._out("employment_tribunal"))
        self.assertIn("Civil Evidence Act 1995 s.1", self._out("civil"))
        self.assertIn("Hearsay and records (arbitration)", self._out("arbitration"))

    def test_s117_is_never_presented_as_the_rule_outside_criminal_proceedings(self):
        for forum in ("employment_tribunal", "civil", "arbitration"):
            out = self._out(forum)
            self.assertNotIn("Business documents (criminal proceedings)", out, forum)
            self.assertNotIn("in the Crown Court a record", out, forum)

    def test_a_passing_mention_of_criminal_in_the_answer_does_not_switch_the_criminal_note_on(self):
        # TRIGGER_STUFFED_ANSWER says "Unlike criminal proceedings ...".
        out = self._out("employment_tribunal")
        self.assertNotIn("Business documents (criminal proceedings)", out)

    def test_unknown_forum_gets_a_note_that_is_true_in_every_forum(self):
        out = self._out("unknown")
        self.assertIn("Hearsay and records depend on the forum", out)
        for rule in ("CJA 2003 s.117", "Employment Tribunal", "Civil Evidence Act 1995 s.1", "Arbitration Act 1996 s.34"):
            self.assertIn(rule, out)

    def test_a_mixed_bundle_gets_each_matching_forum_note(self):
        q = "The Crown Court prosecution runs alongside an Employment Tribunal claim; hearsay in the register?"
        out = append_uk_doctrine_guards("The register is hearsay.", q)
        self.assertIn("Business documents (criminal proceedings)", out)
        self.assertIn("Hearsay and records (Employment Tribunal)", out)

    def test_forum_from_case_exhibits_when_the_question_is_silent(self):
        out = append_uk_doctrine_guards(
            "The rota register is hearsay.",
            "How should the schedules deal with it?",
            "IN THE EMPLOYMENT TRIBUNAL SITTING AT READING",
        )
        self.assertIn("Hearsay and records (Employment Tribunal)", out)

    def test_no_note_when_the_answer_already_covers_the_forum_rule(self):
        answer = "The register is hearsay; the Tribunal is not bound by the strict rules of evidence."
        self.assertEqual(append_uk_doctrine_guards(answer, FORUM_QUESTIONS["employment_tribunal"]), answer)


class DpaPart3GuardTests(unittest.TestCase):
    QUESTIONS = (
        "Can a police force rely on DPA 2018 ss 35-40 when processing for law enforcement purposes?",
        "Covert monitoring of staff by a private employer — lawful under Data Protection Act 2018 ss 35-40?",
        "Data Protection Act 2018 Part 3 — does it apply here?",
    )

    def test_the_note_holds_whoever_the_controller_is(self):
        for q in self.QUESTIONS:
            out = append_uk_doctrine_guards("The controller must consider necessity.", q)
            self.assertIn("Part 3 is for law-enforcement processing only", out, q)
            self.assertIn("Decide the controller and the purpose first", out, q)
            # It must not assert that Part 3 is inapplicable to a controller the question never described.
            self.assertNotIn("does not apply to a private employer", out, q)


class WhistleblowingGuardTests(unittest.TestCase):
    def test_a_stray_one_year_does_not_trigger_the_qualifying_service_note(self):
        q = "She made a protected disclosure to the FCA one year ago and has nine years' service. Is it s.103A?"
        out = append_uk_doctrine_guards("Yes, s.103A applies if the disclosure was the principal reason.", q)
        self.assertNotIn("Burden without qualifying service", out)

    def test_qualifying_service_in_play_triggers_it(self):
        q = "She has less than two years' service and made a protected disclosure; is it s.103A?"
        out = append_uk_doctrine_guards("The s.103A claim is strong.", q)
        self.assertIn("Burden without qualifying service", out)


class NiExtentGuardTests(unittest.TestCase):
    def test_not_duplicated_when_the_answer_already_covers_extent(self):
        q = "I work in Belfast. Does ERA 1996 protect me?"
        answer = "The Employment Rights Act 1996 applies in Great Britain only; in Belfast the 1996 Order governs."
        self.assertEqual(append_uk_doctrine_guards(answer, q), answer)


class RegistryInvariantTests(unittest.TestCase):
    def test_guard_names_are_unique(self):
        names = [name for name, _ in UK_GUARDS]
        self.assertEqual(len(names), len(set(names)))

    def test_every_guard_is_safe_in_every_forum(self):
        # Each guard, in each forum, on an answer stuffed with every trigger word: it returns
        # well-formed bullet notes and never raises.
        for name, guard in UK_GUARDS:
            for forum, question in FORUM_QUESTIONS.items():
                notes = guard(classify_uk_matter(question), question, TRIGGER_STUFFED_ANSWER)
                self.assertIsInstance(notes, list, (name, forum))
                for note in notes:
                    self.assertTrue(note.startswith("- **"), (name, forum, note[:40]))

    def test_no_guard_asserts_a_crown_court_rule_outside_criminal_proceedings(self):
        for name, guard in UK_GUARDS:
            for forum in ("employment_tribunal", "civil", "arbitration"):
                question = FORUM_QUESTIONS[forum]
                for note in guard(classify_uk_matter(question), question, TRIGGER_STUFFED_ANSWER):
                    self.assertNotIn("in the Crown Court a record", note, (name, forum))
                    self.assertNotIn("Business documents (criminal proceedings)", note, (name, forum))


if __name__ == "__main__":
    unittest.main()
