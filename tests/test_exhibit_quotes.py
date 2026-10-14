"""Exhibits as a quotation source (ILL-ARB Whitfield B3: "juggled"/"firefight" for an email that says
"juggling the mezzanine rota" / "firefighting the rota")."""

import unittest

from exhibit_quotes import exhibit_texts, find_exhibit_misquotes, unquote_exhibit_misquotes
from legal_citations import strip_unverified_blockquotes
from legal_verify import audit_legal_draft, build_verifier_feedback

EMAIL = (
    "Rachel, I've seen Callum's email below. Look, I do want to get this right for him, but I need you "
    "to understand what this means operationally – we are still, nearly a year after the first OH "
    "report, juggling the mezzanine rota around him most weeks, and it's the same two or three people "
    "picking up the slack every time. Can we get a proper plan agreed rather than me just firefighting "
    "the rota every week?"
)
MINUTES = (
    "Note: these minutes were prepared from the note-taker's contemporaneous handwritten notes and were "
    "not subsequently circulated to Mr Whitfield for comment or agreement."
)
OH = "time spent on the elevated mezzanine picking floor be reduced where reasonably practicable"
EXHIBITS = [EMAIL, MINUTES, OH]


class FindMisquotesTests(unittest.TestCase):
    def test_the_original_run_misquotes_are_caught_with_the_exhibits_own_words(self):
        text = (
            "The rota was still being “juggled” around Mr Whitfield and Mr Aldred asks for a plan "
            "rather than continuing to “firefight” the rota."
        )
        found = dict(find_exhibit_misquotes(text, EXHIBITS))
        self.assertIn("juggling", found["juggled"])
        self.assertIn("firefighting", found["firefight"])

    def test_exact_quotes_are_not_flagged(self):
        text = (
            "The email says “juggling the mezzanine rota” and the report says "
            "“where reasonably practicable” and the minutes say "
            "“prepared from the note-taker’s contemporaneous handwritten notes”."
        )
        self.assertEqual(find_exhibit_misquotes(text, EXHIBITS), [])

    def test_a_multi_word_quote_with_one_changed_word_is_caught(self):
        text = 'They were "prepared from the note-taker\'s contemporaneous written notes" only.'
        found = find_exhibit_misquotes(text, EXHIBITS)
        self.assertEqual(len(found), 1)
        self.assertIn("handwritten", found[0][1])

    def test_labels_filenames_and_drafted_wording_are_not_claims_about_an_exhibit(self):
        text = (
            'Columns: "Total sought", "unfair dismissal loss" and "single recovery total". '
            'See "DEF-EMAIL-001_Internal-Email-Aldred-Byun.pdf". The respondent should do more than '
            'state "double recovery". Draft: "Claimed once only, under both routes, for the same period."'
        )
        self.assertEqual(find_exhibit_misquotes(text, EXHIBITS), [])

    def test_blockquote_lines_and_html_attributes_are_left_to_their_own_checks(self):
        text = (
            '> *"we are still juggled the mezzanine rota around him"*\n'
            '<a href="https://example.test/?q=juggled" class="legal-ref law" target="_blank">Act, s 1 Law</a>'
        )
        self.assertEqual(find_exhibit_misquotes(text, EXHIBITS), [])

    def test_no_exhibits_means_nothing_to_check(self):
        self.assertEqual(find_exhibit_misquotes('It was "juggled".', []), [])
        self.assertEqual(find_exhibit_misquotes("", EXHIBITS), [])


class UnquoteTests(unittest.TestCase):
    def test_only_the_quote_marks_come_off_and_the_wording_stays(self):
        text = "It was still being “juggled” around him, and “where reasonably practicable” stays quoted."
        out = unquote_exhibit_misquotes(text, EXHIBITS)
        self.assertIn("still being juggled around him", out)
        self.assertIn("“where reasonably practicable”", out)

    def test_html_tags_and_blockquote_lines_are_untouched(self):
        text = (
            '<a href="https://example.test/?q=juggled" class="legal-ref law">Act, s 1 Law</a> said "juggled".\n'
            '> *"juggled"*\n'
        )
        out = unquote_exhibit_misquotes(text, EXHIBITS)
        self.assertIn('href="https://example.test/?q=juggled"', out)
        self.assertIn('said juggled.', out)
        self.assertIn('> *"juggled"*', out)

    def test_nothing_changes_without_exhibits_or_text(self):
        self.assertEqual(unquote_exhibit_misquotes('A "juggled" rota', []), 'A "juggled" rota')
        self.assertEqual(unquote_exhibit_misquotes("", EXHIBITS), "")

    def test_exhibit_texts_reads_the_session_cache_and_never_raises(self):
        class State:
            case_document_cache = {"1": {"text": "abc"}, "2": {"text": "   "}, "3": "junk", "4": {"name": "x"}}

        self.assertEqual(exhibit_texts(State()), ["abc"])
        self.assertEqual(exhibit_texts(None), [])
        self.assertEqual(exhibit_texts(object()), [])


class ExhibitsAsBlockquoteSourceTests(unittest.TestCase):
    QUOTE = (
        "> *“we are still, nearly a year after the first OH report, juggling the mezzanine rota "
        "around him most weeks”*"
    )

    def test_an_exact_exhibit_blockquote_survives_when_exhibits_are_a_source(self):
        out = strip_unverified_blockquotes(self.QUOTE, [], extra_texts=EXHIBITS)
        self.assertEqual(out, self.QUOTE)

    def test_it_is_still_stripped_when_no_source_holds_it(self):
        out = strip_unverified_blockquotes(self.QUOTE, [])
        self.assertIn("Unverified quotation removed", out)

    def test_an_invented_blockquote_is_stripped_even_with_exhibits(self):
        invented = "> *“Mr Aldred confirmed that every adjustment was put in writing on the date requested”*"
        out = strip_unverified_blockquotes(invented, [], extra_texts=EXHIBITS)
        self.assertIn("Unverified quotation removed", out)


class VerifierTests(unittest.TestCase):
    DRAFT = "The rota was still being “juggled” around him."

    def test_a_misquote_is_reported_and_tells_the_model_what_the_exhibit_says(self):
        report = audit_legal_draft(self.DRAFT, [], extra_quote_sources=EXHIBITS)
        self.assertEqual(len(report.exhibit_misquotes), 1)
        self.assertTrue(report.has_issues)
        self.assertIn("not verbatim to the exhibit", report.summary())
        feedback = build_verifier_feedback(report)
        self.assertIn("the exhibit says", feedback)
        self.assertIn("juggling", feedback)
        self.assertIn("paraphrase WITHOUT quote marks", feedback)

    def test_no_report_without_exhibits(self):
        report = audit_legal_draft(self.DRAFT, [])
        self.assertEqual(report.exhibit_misquotes, [])


if __name__ == "__main__":
    unittest.main()
