"""C1 (critical, "nothing fabricated"): an answer weighed a witness statement against "Solway's
suggestion that Cairngorm had no FAT evidence at all" - a position no document records.

The sentences that attribute a statement or position to a party, witness or document are picked out
deterministically; a judge (faked here) reads them against the exhibits by MEANING. It fails open and
is skipped without exhibits or a judge."""

import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from attribution_check import attribution_check_enabled, candidate_attributions, check_attributions


def setUpModule():
    # The check is off by default (see attribution_check_enabled); these tests exercise it switched on.
    os.environ["LEGAL_ATTRIBUTION_CHECK"] = "true"


def tearDownModule():
    os.environ.pop("LEGAL_ATTRIBUTION_CHECK", None)
from legal_verify import audit_legal_draft, build_verifier_feedback, make_legal_verifier

C1_SENTENCE = (
    "Ms MacRae's statement is more probative than Solway's suggestion that Cairngorm had no FAT "
    "evidence at all."
)
EXHIBITS = ["Solway's NCR records that the FAT records were not held by Solway.", "Ferries: I have not seen the FAT records."]


def fake_judge(unsupported_numbers, why="no document records this"):
    calls = []

    def judge(system, user):
        calls.append((system, user))
        return json.dumps({"unsupported": [{"n": n, "why": why} for n in unsupported_numbers]})

    judge.calls = calls
    return judge


class DefaultOnTests(unittest.TestCase):
    def test_it_is_on_by_default_and_can_be_switched_off(self):
        # Enabled after a probe with known-false attributions among genuine ones: 5/5 caught and 0 genuine
        # flagged across 3 repeats x 2 passes (earlier designs: 60% and then 12% of genuine ones flagged).
        env = {k: v for k, v in os.environ.items() if k != "LEGAL_ATTRIBUTION_CHECK"}
        with patch.dict(os.environ, env, clear=True):
            self.assertTrue(attribution_check_enabled())
        for off in ("false", "0", "no", "off", "FALSE"):
            with patch.dict(os.environ, {"LEGAL_ATTRIBUTION_CHECK": off}):
                self.assertFalse(attribution_check_enabled(), off)
                judge = fake_judge([1])
                self.assertEqual(check_attributions(C1_SENTENCE, EXHIBITS, judge), [])
                self.assertEqual(judge.calls, [])


class CandidateTests(unittest.TestCase):
    def test_the_c1_sentence_and_ordinary_attributions_are_candidates(self):
        text = f"{C1_SENTENCE} Mr Aldred says that adjustments were fully implemented from April 2023."
        found = candidate_attributions(text)
        self.assertEqual(len(found), 2)
        self.assertIn("Solway's suggestion that Cairngorm had no FAT evidence at all", found[0])

    def test_the_answers_own_hypotheses_and_drafting_are_not_attributions(self):
        text = (
            "The Respondent may argue that the loss ended in January 2025, and the schedule should state that "
            "the figures are provisional. Mr Aldred could say that the role was never offered to him."
        )
        self.assertEqual(candidate_attributions(text), [])

    def test_the_live_false_positives_are_not_candidates(self):
        # The three sentences the judge flagged on the live run (advice and a schedule line item, not
        # reports of what anyone said). Sending the model back over these risks cut or hedged content.
        live_false_positives = (
            "If the Claimant alleges that the failure to make adjustments caused loss before termination, that loss should be separately identified.",
            "This format is particularly useful where the Claimant says the Tribunal could make different findings on the various claims.",
            "- the net pay the Claimant says he would have received from Brightwell;",
        )
        for sentence in live_false_positives:
            self.assertEqual(candidate_attributions(sentence), [], sentence)

    def test_real_reports_of_what_someone_said_are_still_candidates(self):
        for sentence in (
            "The Claimant says that he found part-time work with another employer from January 2025 at lower pay.",
            "The Respondent denies that the criteria were discriminatory and says that they reflected an operational need.",
            "Mr Aldred says that adjustments were fully implemented from April 2023 onwards.",
            C1_SENTENCE,
        ):
            self.assertEqual(len(candidate_attributions(sentence)), 1, sentence)

    def test_conditional_openings_and_inner_conditionals_are_skipped(self):
        for sentence in (
            "Where the Respondent says that the role was never offered, it should produce the job description.",
            "Whether the Claimant argues that the loss continues is a matter for the findings of fact.",
            "The schedule helps when the Respondent contends that the loss ended in January 2025 for another reason.",
        ):
            self.assertEqual(candidate_attributions(sentence), [], sentence)

    def test_the_judge_is_told_to_ignore_advice_conditionals_and_schedule_wording(self):
        judge = fake_judge([])
        check_attributions(C1_SENTENCE, EXHIBITS, judge)
        user = judge.calls[0][1]
        self.assertIn("'if the Claimant alleges...'", user)
        self.assertIn("what a schedule or document should contain", user)
        self.assertIn("Only judge a sentence that reports what someone HAS said", user)

    def test_quotations_tables_and_headings_are_not_claims(self):
        text = (
            "> Mr Aldred says that adjustments were fully implemented from April 2023 onwards, he states.\n"
            "| Mr Aldred says that adjustments were fully implemented | CLM-WIT-001 |\n"
            "## Mr Aldred says that adjustments were fully implemented from April 2023 onwards\n"
        )
        self.assertEqual(candidate_attributions(text), [])

    def test_links_and_markdown_are_stripped_from_the_sentence(self):
        text = "**Mr Aldred** says that adjustments were [fully implemented](https://x.test/a) from April 2023 onwards."
        self.assertEqual(
            candidate_attributions(text), ["Mr Aldred says that adjustments were fully implemented from April 2023 onwards."]
        )

    def test_the_number_of_claims_checked_is_bounded(self):
        text = "\n".join(f"Mr Aldred says that point number {i} was agreed between the parties in writing." for i in range(80))
        self.assertEqual(len(candidate_attributions(text)), 40)


class CheckTests(unittest.TestCase):
    def test_the_unsupported_attribution_is_returned_with_the_judges_reason(self):
        judge = fake_judge([1], why="no document records Solway suggesting this")
        out = check_attributions(f"{C1_SENTENCE}", EXHIBITS, judge)
        self.assertEqual(len(out), 1)
        self.assertIn("Solway's suggestion", out[0][0])
        self.assertIn("no document records", out[0][1])

    def test_the_judge_sees_the_exhibits_and_the_numbered_claims_and_is_told_to_resolve_doubt_as_supported(self):
        judge = fake_judge([])
        check_attributions(C1_SENTENCE, EXHIBITS, judge)
        _system, user = judge.calls[0]
        self.assertIn("the FAT records were not held by Solway", user)  # its own excerpt, not the whole bundle
        # the judge is asked about the attribution itself, not the answer's own weighing around it
        self.assertIn("CLAIM 1: Solway's suggestion that Cairngorm had no FAT evidence at all.", user)
        self.assertNotIn("more probative than", user)
        self.assertIn("If you are unsure, use 'supported'", user)

    def test_it_fails_open_on_an_error_a_bad_reply_or_an_out_of_range_number(self):
        def boom(system, user):
            raise RuntimeError("timeout")

        self.assertEqual(check_attributions(C1_SENTENCE, EXHIBITS, boom), [])
        self.assertEqual(check_attributions(C1_SENTENCE, EXHIBITS, lambda s, u: "not json at all"), [])
        self.assertEqual(check_attributions(C1_SENTENCE, EXHIBITS, fake_judge([7])), [])

    def test_it_reads_a_fenced_json_reply(self):
        judge = lambda s, u: '```json\n{"unsupported": [{"n": 1, "why": "x"}]}\n```'  # noqa: E731
        self.assertEqual(len(check_attributions(C1_SENTENCE, EXHIBITS, judge)), 1)

    def test_it_is_skipped_without_exhibits_a_judge_or_candidates_and_can_be_switched_off(self):
        judge = fake_judge([1])
        self.assertEqual(check_attributions(C1_SENTENCE, [], judge), [])
        self.assertEqual(check_attributions(C1_SENTENCE, EXHIBITS, None), [])
        self.assertEqual(check_attributions("Nothing is attributed here at all.", EXHIBITS, judge), [])
        with patch.dict(os.environ, {"LEGAL_ATTRIBUTION_CHECK": "false"}):
            self.assertEqual(check_attributions(C1_SENTENCE, EXHIBITS, judge), [])
        self.assertEqual(judge.calls, [])


def verdicts(*rows):
    """A judge reply in the current shape: rows of (n, verdict, quote, why)."""
    return json.dumps({"results": [{"n": n, "verdict": v, "quote": q, "why": w} for n, v, q, w in rows]})


GENUINE = "Mr Aldred says that adjustments were fully implemented from April 2023 onwards and remained in place throughout his employment."
ALDRED_STATEMENT = (
    "Document ID: DEF-WIT-001 Document Type: Witness Statement Author: Steven Aldred (Operations Manager)\n\n"
    "2. Following the OH report in April 2023, I adjusted duties. These adjustments were fully implemented "
    "from April 2023 onwards and remained in place throughout his employment."
)
OKORO_STATEMENT = (
    "Document ID: CLM-WIT-002 Document Type: Witness Statement Author: Denise Okoro (Supervisor, Swindon DC)\n\n"
    "5. I have not seen a written job description for the Site Coordinator role."
)


class EvidenceGroundedJudgeTests(unittest.TestCase):
    """A whole-bundle judge flagged ~60% of GENUINE attributions in a live probe (5 of 8, near-verbatim
    ones included). Each claim is now judged against its own excerpts, with code-side overrides."""

    def test_each_claim_is_shown_with_its_own_excerpts_headed_by_the_documents_author_line(self):
        text = f"{GENUINE}\n\nMs Okoro says that she has not seen a written job description for the Site Coordinator role."
        judge = fake_judge([])
        check_attributions(text, [ALDRED_STATEMENT, OKORO_STATEMENT], judge)
        user = judge.calls[0][1]
        claim1, claim2 = user.split("CLAIM 2:")
        self.assertIn("Author: Steven Aldred", claim1)  # the Aldred claim sees Aldred's statement and header
        self.assertIn("adjustments were fully implemented", claim1)
        # only relevant documents are shown: each claim sees its own author's statement, not the other's
        self.assertNotIn("Author: Denise Okoro", claim1)
        self.assertIn("Author: Denise Okoro", claim2)
        self.assertNotIn("Author: Steven Aldred", claim2)

    def test_the_best_passage_of_several_documents_is_shown_not_many_windows_of_one(self):
        # Live miss: overlapping windows from one paragraph filled every slot, so the Claimant's own
        # statement (which supported the claim) was never shown and the claim was judged unsupported.
        aldred = ("Document Type: Witness Statement Author: Steven Aldred\n\n"
                  "6. At the meeting I explained the criteria. He did not directly say the criteria discriminated "
                  "because of his back condition. He asked why his score was lower. I recorded his back condition.")
        oh = "Document Type: OH report\n\nHis back condition is long-standing and has a substantial adverse effect."
        claimant = ("Document Type: Witness Statement Author: Callum Whitfield (Claimant)\n\n"
                    "6. At the meeting I specifically challenged the criterion: that's because of my back.")
        judge = fake_judge([])
        check_attributions(
            "The Claimant says that he expressly connected the issue to his back condition at that meeting.",
            [aldred, oh, claimant],
            judge,
        )
        user = judge.calls[0][1]
        self.assertIn("I specifically challenged the criterion", user)
        self.assertIn("document 1:", user)
        self.assertIn("document 3:", user)

    def test_only_an_unsupported_verdict_is_flagged(self):
        judge = lambda s, u: verdicts(  # noqa: E731
            (1, "supported", "adjustments were fully implemented", ""),
            (2, "not_an_attribution", "", "advice"),
            (3, "unsupported", "", "no document records this"),
        )
        text = (
            f"{GENUINE}\n\nMs Okoro says that she has not seen a written job description for the Site Coordinator role.\n\n"
            f"{C1_SENTENCE}"
        )
        out = check_attributions(text, [ALDRED_STATEMENT, OKORO_STATEMENT], judge)
        self.assertEqual([s[:20] for s, _ in out], [C1_SENTENCE[:20]])

    def test_a_flag_on_a_near_verbatim_claim_is_dropped_whatever_the_judge_says(self):
        # The probe's failure: the judge called a near-word-for-word statement unsupported.
        judge = lambda s, u: verdicts((1, "unsupported", "", "no document records this"))  # noqa: E731
        self.assertEqual(check_attributions(GENUINE, [ALDRED_STATEMENT], judge), [])

    def test_a_flag_that_comes_with_a_quote_of_support_is_contradictory_so_dropped(self):
        judge = lambda s, u: verdicts((1, "unsupported", "I have not seen a written job description", "x"))  # noqa: E731
        claim = "Ms Okoro says that she has seen no written job description for the new Site Coordinator post at Reading."
        self.assertEqual(check_attributions(claim, [OKORO_STATEMENT], judge), [])

    def test_a_genuinely_unsupported_claim_is_still_flagged(self):
        judge = lambda s, u: verdicts((1, "unsupported", "", "no document records Ms Okoro attending"))  # noqa: E731
        claim = "Ms Okoro says that she attended the 22 July 2024 meeting as Mr Whitfield's companion."
        out = check_attributions(claim, [ALDRED_STATEMENT, OKORO_STATEMENT], judge)
        self.assertEqual(len(out), 1)
        self.assertIn("no document records Ms Okoro attending", out[0][1])

    def test_an_unsupported_verdict_with_no_excerpt_at_all_is_still_reported_not_hidden(self):
        judge = fake_judge([])
        check_attributions(C1_SENTENCE, ["Completely unrelated text about lorries."], judge)
        self.assertIn("no excerpt of any case document is relevant to this claim", judge.calls[0][1])


class TwoPassTests(unittest.TestCase):
    """A reasoning judge is not deterministic: a probe missed the exact C1 shape in 2 of 3 calls. C1 is
    the critical failure, so the judge runs twice and a claim is reported if EITHER pass flags it."""

    def _scripted(self, *replies):
        calls = []

        def judge(system, user):
            calls.append(user)
            reply = replies[min(len(calls), len(replies)) - 1]
            if isinstance(reply, Exception):
                raise reply
            return reply

        judge.calls = calls
        return judge

    def test_a_claim_flagged_in_only_one_pass_is_reported(self):
        judge = self._scripted(
            verdicts((1, "supported", "x", "")),
            verdicts((1, "unsupported", "", "no document records Solway suggesting this")),
        )
        out = check_attributions(C1_SENTENCE, EXHIBITS, judge)
        self.assertEqual(len(judge.calls), 2)
        self.assertEqual(len(out), 1)
        self.assertIn("no document records Solway", out[0][1])

    def test_the_reported_sentence_is_the_whole_sentence_but_the_judge_saw_only_the_attribution(self):
        judge = self._scripted(verdicts((1, "unsupported", "", "w")))
        out = check_attributions(C1_SENTENCE, EXHIBITS, judge)
        self.assertEqual(out[0][0], C1_SENTENCE)  # the model is shown the sentence it must fix
        self.assertNotIn("Ms MacRae", judge.calls[0])  # the judge is not shown the answer's own weighing

    def test_a_claim_flagged_in_both_passes_is_reported_once(self):
        judge = self._scripted(verdicts((1, "unsupported", "", "a")), verdicts((1, "unsupported", "", "b")))
        self.assertEqual(len(check_attributions(C1_SENTENCE, EXHIBITS, judge)), 1)

    def test_a_pass_that_errors_does_not_hide_what_the_other_found(self):
        judge = self._scripted(RuntimeError("timeout"), verdicts((1, "unsupported", "", "w")))
        self.assertEqual(len(check_attributions(C1_SENTENCE, EXHIBITS, judge)), 1)

    def test_when_every_pass_errors_nothing_is_reported(self):
        judge = self._scripted(RuntimeError("a"), RuntimeError("b"))
        self.assertEqual(check_attributions(C1_SENTENCE, EXHIBITS, judge), [])

    def test_the_number_of_passes_can_be_set(self):
        judge = self._scripted(verdicts((1, "supported", "x", "")))
        with patch.dict(os.environ, {"LEGAL_ATTRIBUTION_PASSES": "1"}):
            check_attributions(C1_SENTENCE, EXHIBITS, judge)
        self.assertEqual(len(judge.calls), 1)

    def test_the_trace_records_every_verdict_for_debugging(self):
        judge = self._scripted(verdicts((1, "unsupported", "", "no record")))
        trace = []
        check_attributions(C1_SENTENCE, EXHIBITS, judge, trace=trace)
        self.assertEqual([t["pass"] for t in trace], [1, 2])  # one claim, judged once per pass
        self.assertEqual(trace[0]["verdict"], "unsupported")
        self.assertIn("Solway's suggestion", trace[0]["claim"])

    def test_the_clause_is_cut_from_the_subject_in_a_compound_sentence(self):
        sentence = (
            "The minutes record that the Claimant asked why his score was lower, while the Claimant says that he "
            "expressly connected the issue to his back condition."
        )
        judge = fake_judge([])
        check_attributions(sentence, EXHIBITS, judge)
        self.assertIn("CLAIM 1: the Claimant says that he expressly connected the issue", judge.calls[0][1])
        self.assertNotIn("The minutes record", judge.calls[0][1].split("EXCERPTS:")[0])


class VerifierIntegrationTests(unittest.TestCase):
    def test_the_audit_reports_it_and_the_feedback_tells_the_model_to_delete_or_recast_it(self):
        report = audit_legal_draft(C1_SENTENCE, [], uk=True, extra_quote_sources=EXHIBITS, attribution_judge=fake_judge([1]))
        self.assertTrue(report.has_issues)
        self.assertIn("attribution(s) no case document supports", report.summary())
        feedback = build_verifier_feedback(report)
        self.assertIn("Solway's suggestion", feedback)
        self.assertIn("Never attribute to a party, witness or document what its own text does not say", feedback)
        self.assertIn("recast it as your own inference", feedback)

    def test_not_checked_outside_uk_mode_or_without_exhibits(self):
        judge = fake_judge([1])
        self.assertEqual(audit_legal_draft(C1_SENTENCE, [], uk=False, extra_quote_sources=EXHIBITS, attribution_judge=judge).unsupported_attributions, [])
        self.assertEqual(audit_legal_draft(C1_SENTENCE, [], uk=True, attribution_judge=judge).unsupported_attributions, [])
        self.assertEqual(judge.calls, [])

    def test_the_paid_judge_call_is_not_made_on_the_final_pass(self):
        state = SimpleNamespace(
            last_search_legal_results=[],
            case_document_manifest=[],
            case_document_cache={"1": {"text": EXHIBITS[0]}},
        )
        judge = fake_judge([1])
        verify = make_legal_verifier(state, max_rounds=1, query="A question?", uk=True, judge=judge)
        first = verify(C1_SENTENCE, 0)  # a revise round remains: the judge runs and the model is told
        self.assertIsNotNone(first)
        self.assertIn("Solway's suggestion", first.message)
        self.assertEqual(len(judge.calls), 2)  # two passes (see TwoPassTests)
        self.assertIsNone(verify(C1_SENTENCE, 1))  # final pass: no feedback possible, so no paid call
        self.assertEqual(len(judge.calls), 2)


if __name__ == "__main__":
    unittest.main()
