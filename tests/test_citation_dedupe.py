"""Identical repeated citation links are removed; a different provision of the same document is not.

The ILL-ARB Whitfield response linked ERA 1996 s 123 five times and Chagger three times, and the
prompt rule against repeats was only partly obeyed on later runs, so the finalizer enforces it.
"""

import unittest

from legal_citations import dedupe_repeated_citation_links


def law(label, href):
    return f'<a href="{href}" class="legal-ref law" target="_blank">{label} Law</a>'


def case(label, href):
    return f'<a href="{href}" class="legal-ref jurisprudence" target="_blank">{label} Jurisprudence</a>'


ERA123 = law("Employment Rights Act 1996, s 123", "https://www.legislation.gov.uk/ukpga/1996/18/section/123")
ERA118 = law("Employment Rights Act 1996, s 118", "https://www.legislation.gov.uk/ukpga/1996/18/section/118")
EQA124 = law("Equality Act 2010, s 124", "https://www.legislation.gov.uk/ukpga/2010/15/section/124")
EQA119 = law("Equality Act 2010, s 119", "https://www.legislation.gov.uk/ukpga/2010/15/section/119")
CHAGGER = case("*Chagger v Abbey National plc* [2009] EWCA Civ 1202", "https://caselaw.nationalarchives.gov.uk/ewca/civ/2009/1202")


class DedupeTests(unittest.TestCase):
    def test_first_occurrence_stays_and_an_inline_repeat_becomes_plain_words(self):
        text = f"Loss is assessed under {ERA123}. Mitigation also falls under {ERA123} in this case."
        out = dedupe_repeated_citation_links(text)
        self.assertEqual(out.count("<a href"), 1)
        self.assertIn("Employment Rights Act 1996, s 123 in this case", out)

    def test_a_repeat_in_a_trailing_trail_is_removed_with_its_separator(self):
        text = f"Para one. {ERA123}; {CHAGGER}\n\nPara two. {EQA124}; {ERA123}; {EQA119}\n"
        out = dedupe_repeated_citation_links(text)
        self.assertEqual(out.count(ERA123), 1)
        self.assertIn(f"{EQA124}; {EQA119}", out)
        self.assertNotIn("; ;", out)

    def test_a_closing_list_that_only_repeats_earlier_citations_disappears(self):
        text = f"Body {ERA123} and {CHAGGER} are explained here.\n\nBottom line.\n{ERA123}; {CHAGGER}\n"
        out = dedupe_repeated_citation_links(text)
        self.assertEqual(out.count("<a href"), 2)
        self.assertNotIn("\n;", out)
        self.assertTrue(out.rstrip().endswith("Bottom line."))

    def test_a_different_provision_of_the_same_document_is_not_a_repeat(self):
        text = f"See {EQA119} then {EQA124}; later {ERA118} and {ERA123}."
        self.assertEqual(dedupe_repeated_citation_links(text), text)

    def test_same_label_with_a_different_href_is_kept(self):
        other = law("Employment Rights Act 1996, s 123", "https://www.legislation.gov.uk/ukpga/1996/18/section/123/enacted")
        text = f"First {ERA123}. Second {other}."
        self.assertEqual(dedupe_repeated_citation_links(text), text)

    def test_a_repeated_blockquote_attribution_line_is_deleted_whole(self):
        text = f"> *\"quoted\"*\n> - {CHAGGER}\n\nLater.\n\n> *\"another\"*\n> - {CHAGGER}\n\nEnd."
        out = dedupe_repeated_citation_links(text)
        self.assertEqual(out.count(CHAGGER), 1)
        self.assertNotIn("> - \n", out)
        self.assertTrue(out.endswith("End."))

    def test_a_repeat_that_is_the_object_of_its_sentence_keeps_its_words(self):
        # Deleting it would leave "It also follows from." — keep the words, drop only the link.
        text = f"Mitigation is required. It follows from {ERA123}\n\nIt also follows from {ERA123}.\n"
        out = dedupe_repeated_citation_links(text)
        self.assertEqual(out.count("<a href"), 1)
        self.assertIn("It also follows from Employment Rights Act 1996, s 123.", out)

    def test_a_repeat_after_a_complete_sentence_is_deleted_and_the_sentence_stays_whole(self):
        text = f"Loss is assessed once. {ERA123}\n\nMitigation is required. {ERA123}\n"
        out = dedupe_repeated_citation_links(text)
        self.assertEqual(out.count("<a href"), 1)
        self.assertIn("Mitigation is required.", out)

    def test_idempotent_and_leaves_plain_text_alone(self):
        text = f"A {ERA123}; {EQA124}. B {ERA123}; {EQA124}."
        once = dedupe_repeated_citation_links(text)
        self.assertEqual(dedupe_repeated_citation_links(once), once)
        self.assertEqual(dedupe_repeated_citation_links("No citations here."), "No citations here.")
        self.assertEqual(dedupe_repeated_citation_links(""), "")


if __name__ == "__main__":
    unittest.main()
