"""Two ILL-ARB follow-ups found on live runs.

1. The blockquote gate deleted the model's own proposed schedule wording ("The Claimant could
   include wording such as: > ...") as an "unverified quotation" — 14-15 visible notices per answer.
   For UK answers only a blockquote formatted as a quotation (opens with a quote mark) claims a source.
2. The unfair dismissal compensatory award is capped and Equality Act compensation is not (ERA 1996
   s.124 / s.126); no answer to a combined schedule-of-loss question said so.
"""

import unittest

from legal_citations import quotation_blockquote_lines, strip_unverified_blockquotes
from legal_fact_boost import UK_GUARDS, append_uk_doctrine_guards
from legal_verify import audit_legal_draft, build_verifier_feedback, missing_cap_statement

DRAFT_WORDING = (
    "The Claimant could include wording such as:\n"
    "> The Claimant relies on unfair dismissal and Equality Act claims. Each overlapping loss is claimed once only.\n"
    "> References to more than one cause of action identify alternative bases for the same item.\n"
)
QUOTED_BLOCK = (
    "> *“The tribunal must award such sum as it considers just and\n"
    "> equitable in all the circumstances having regard to the loss.”*\n"
    "> - [Employment Rights Act 1996, s 123 Law](https://www.legislation.gov.uk/ukpga/1996/18/section/123)\n"
)
EXHIBIT = (
    "we are still, nearly a year after the first OH report, juggling the mezzanine rota around him most weeks"
)


class DraftWordingTests(unittest.TestCase):
    def test_uk_mode_leaves_proposed_wording_blockquotes_alone(self):
        out = strip_unverified_blockquotes(DRAFT_WORDING, [], quotation_blocks_only=True)
        self.assertEqual(out, DRAFT_WORDING)

    def test_default_mode_is_unchanged_and_still_strips_every_unverified_blockquote_line(self):
        out = strip_unverified_blockquotes(DRAFT_WORDING, [])
        self.assertEqual(out.count("Unverified quotation removed"), 2)

    def test_an_unverified_quotation_block_is_removed_whole_with_no_orphan_continuation(self):
        out = strip_unverified_blockquotes(QUOTED_BLOCK, [], quotation_blocks_only=True)
        self.assertEqual(out.count("Unverified quotation removed"), 2)
        self.assertNotIn("equitable in all the circumstances", out)
        self.assertIn("[Employment Rights Act 1996, s 123 Law]", out)  # the attribution line stays

    def test_an_exact_exhibit_quotation_survives(self):
        quote = f"> *“{EXHIBIT}”*"
        out = strip_unverified_blockquotes(quote, [], extra_texts=[EXHIBIT], quotation_blocks_only=True)
        self.assertEqual(out, quote)

    def test_a_blank_line_ends_a_quotation_block(self):
        text = (
            "> “A long quoted sentence that is comfortably over the length threshold”\n\n"
            "> unquoted proposal text that is long enough to count as a line of wording\n"
        )
        spans = quotation_blockquote_lines(text)
        self.assertEqual(len(spans), 1)
        self.assertIn("A long quoted sentence", spans[0][2])

    def test_the_audit_ignores_proposed_wording_in_uk_mode_only(self):
        self.assertEqual(audit_legal_draft(DRAFT_WORDING, [], uk=True).unverified_quotes, [])
        self.assertEqual(len(audit_legal_draft(DRAFT_WORDING, [], uk=False).unverified_quotes), 2)

    def test_the_audit_still_reports_an_unverified_quotation_in_uk_mode(self):
        self.assertTrue(audit_legal_draft(QUOTED_BLOCK, [], uk=True).unverified_quotes)


REMEDIES_Q = "How should the schedules of loss and counter-schedules avoid double recovery?"
DRAFT_NO_CAP = "The unfair dismissal compensatory award and Equality Act compensation overlap; claim each loss once."


class CapCheckTests(unittest.TestCase):
    def test_flags_a_remedies_draft_that_never_states_the_cap(self):
        self.assertTrue(missing_cap_statement(REMEDIES_Q, DRAFT_NO_CAP))

    def test_not_flagged_once_the_cap_is_stated(self):
        for stated in (
            "The compensatory award is capped; Equality Act compensation is not.",
            "The compensatory award is subject to a statutory maximum.",
            "The compensatory award cannot exceed one year's gross pay.",
        ):
            self.assertFalse(missing_cap_statement(REMEDIES_Q, stated), stated)

    def test_capacity_and_capable_are_not_a_cap(self):
        self.assertTrue(missing_cap_statement(REMEDIES_Q, DRAFT_NO_CAP + " He had capacity for full-time work."))

    def test_not_flagged_for_a_question_that_is_not_about_remedies(self):
        self.assertFalse(missing_cap_statement("Was the dismissal procedurally fair?", DRAFT_NO_CAP))

    def test_not_flagged_when_the_draft_never_discusses_the_compensatory_award(self):
        self.assertFalse(missing_cap_statement(REMEDIES_Q, "Quantum depends on the findings of fact."))

    def test_the_verifier_sends_the_model_to_fetch_s124_and_never_to_invent_a_figure(self):
        report = audit_legal_draft(DRAFT_NO_CAP, [], uk=True, question=REMEDIES_Q)
        self.assertTrue(report.missing_cap)
        self.assertTrue(report.has_issues)
        self.assertIn("compensatory award cap not stated", report.summary())
        feedback = build_verifier_feedback(report)
        self.assertIn("s.124", feedback)
        self.assertIn("Do not state a cap figure you have not fetched", feedback)

    def test_not_checked_outside_uk_mode(self):
        self.assertFalse(audit_legal_draft(DRAFT_NO_CAP, [], uk=False, question=REMEDIES_Q).missing_cap)


class CapGuardTests(unittest.TestCase):
    def test_the_backstop_note_is_appended_when_the_answer_never_says_it(self):
        out = append_uk_doctrine_guards(DRAFT_NO_CAP, REMEDIES_Q)
        self.assertIn("Unfair dismissal cap v Equality Act compensation", out)
        self.assertIn("ERA 1996 s.124", out)
        self.assertIn("no statutory cap", out)
        self.assertIn("ERA 1996 s.126", out)

    def test_the_note_states_no_figure(self):
        note = append_uk_doctrine_guards(DRAFT_NO_CAP, REMEDIES_Q).split("Additional points", 1)[1]
        self.assertNotIn("£", note)
        self.assertNotIn("123,543", note)

    def test_no_note_when_the_answer_already_states_the_cap(self):
        answer = DRAFT_NO_CAP + " The compensatory award is capped; the Equality Act award is uncapped."
        self.assertEqual(append_uk_doctrine_guards(answer, REMEDIES_Q), answer)

    def test_no_note_without_an_equality_act_claim(self):
        self.assertEqual(
            append_uk_doctrine_guards("The compensatory award reflects loss.", "What is my unfair dismissal worth?"),
            "The compensatory award reflects loss.",
        )

    def test_it_is_registered_so_the_forum_invariants_cover_it(self):
        self.assertIn("unfair_dismissal_cap", [name for name, _ in UK_GUARDS])


if __name__ == "__main__":
    unittest.main()
