"""Two ILL-ARB follow-ups found on live runs.

1. The blockquote gate deleted the model's own proposed schedule wording ("The Claimant could
   include wording such as: > ...") as an "unverified quotation" — 14-15 visible notices per answer.
   For UK answers only a blockquote formatted as a quotation (opens with a quote mark) claims a source.
2. The unfair dismissal compensatory award is capped and Equality Act compensation is not (ERA 1996
   s.124 / s.126); no answer to a combined schedule-of-loss question said so.
"""

import unittest

from legal_citations import quotation_blockquote_lines, strip_unverified_blockquotes
from legal_fact_boost import REQUIRED_PROVISIONS, UK_GUARDS, append_uk_doctrine_guards, missing_required_provisions
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

    def test_a_list_item_that_merely_mentions_a_cap_does_not_state_it(self):
        # Live run: the only mention was "5. any statutory cap or other limitation, where relevant" —
        # the check wrongly counted it and neither the verifier nor the guard stepped in.
        listed = DRAFT_NO_CAP + "\nThe schedule should show: 4. interest; and 5. any statutory cap or other limitation, where relevant."
        self.assertTrue(missing_cap_statement(REMEDIES_Q, listed))
        self.assertIn("Unfair dismissal cap v Equality Act compensation", append_uk_doctrine_guards(listed, REMEDIES_Q))

    def test_citing_the_cap_provision_counts_as_stating_it(self):
        for stated in (
            "The compensatory award is limited by section 124 of the Employment Rights Act 1996.",
            "The compensatory award is limited: [Employment Rights Act 1996, s 124 Law](https://x.test/124).",
            "The compensatory award is limited by s.124(1ZA).",
        ):
            self.assertFalse(missing_cap_statement(REMEDIES_Q, stated), stated)

    def test_the_equality_act_s124_is_not_the_unfair_dismissal_cap(self):
        # Equality Act 2010 s.124 is the tribunal's remedies power, not the ERA s.124 cap.
        eqa = DRAFT_NO_CAP + " Under section 124 of the Equality Act 2010 the Tribunal may award compensation."
        self.assertTrue(missing_cap_statement(REMEDIES_Q, eqa))

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
        self.assertIn("controlling provision(s) not stated", report.summary())
        self.assertIn("era_s124_cap", report.summary())
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
        answer = (
            DRAFT_NO_CAP
            + " The compensatory award is capped; the Equality Act award is uncapped."
            + " Under ERA 1996 s.126 compensation cannot be awarded twice for the same loss."
        )
        self.assertEqual(append_uk_doctrine_guards(answer, REMEDIES_Q), answer)

    def test_no_note_without_an_equality_act_claim(self):
        self.assertEqual(
            append_uk_doctrine_guards("The compensatory award reflects loss.", "What is my unfair dismissal worth?"),
            "The compensatory award reflects loss.",
        )

    def test_it_is_registered_so_the_forum_invariants_cover_it(self):
        self.assertIn("required_provisions", [name for name, _ in UK_GUARDS])


class RequiredProvisionRegistryTests(unittest.TestCase):
    """E2 ("law and procedure correct"): the answer omitted both the cap and the bar on double
    compensation. Controlling provisions are a registry, so a new one is one entry."""

    OVERLAP_DRAFT = "Unfair dismissal and Equality Act compensation overlap; the same loss must not be claimed twice."

    def _names(self, question, draft):
        return [p.name for p in missing_required_provisions(question, draft)]

    def test_s126_is_required_when_unfair_dismissal_and_equality_act_loss_overlap(self):
        self.assertIn("era_s126_no_double_compensation", self._names(REMEDIES_Q, self.OVERLAP_DRAFT))

    def test_s126_is_stated_by_citation_or_by_its_rule(self):
        for stated in (
            self.OVERLAP_DRAFT + " See ERA 1996 s.126.",
            self.OVERLAP_DRAFT + " [Employment Rights Act 1996, s 126 Law](https://x.test/126)",
            self.OVERLAP_DRAFT + " Section 126 of the Employment Rights Act bars it.",
            self.OVERLAP_DRAFT + " A tribunal may not award compensation under either Act for the same loss.",
        ):
            self.assertNotIn("era_s126_no_double_compensation", self._names(REMEDIES_Q, stated), stated)

    def test_s126_is_not_required_without_an_overlap_between_the_two_regimes(self):
        self.assertEqual(self._names("What is my claim worth?", "Unfair dismissal loss is assessed on the facts."), [])
        self.assertNotIn(
            "era_s126_no_double_compensation",
            self._names(REMEDIES_Q, "The Equality Act allows injury to feelings. Quantum depends on the findings."),
        )

    def test_the_guard_appends_a_note_for_each_provision_still_missing_and_only_those(self):
        out = append_uk_doctrine_guards(DRAFT_NO_CAP + " See ERA 1996 s.126.", REMEDIES_Q)
        self.assertIn("Unfair dismissal cap v Equality Act compensation", out)
        self.assertNotIn("No double compensation (ERA 1996 s.126)", out)
        out = append_uk_doctrine_guards(DRAFT_NO_CAP + " The award is capped by section 124 of the Employment Rights Act.", REMEDIES_Q)
        self.assertIn("No double compensation (ERA 1996 s.126)", out)
        self.assertNotIn("Unfair dismissal cap v Equality Act compensation", out)

    def test_every_provision_carries_a_fetch_hint_and_a_note(self):
        for p in REQUIRED_PROVISIONS:
            self.assertTrue(p.fetch_hint and p.note and p.name, p.name)


if __name__ == "__main__":
    unittest.main()
