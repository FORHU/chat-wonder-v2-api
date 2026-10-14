"""The legal persona's prompt must agree with how generated documents are delivered: the drafted
text is saved as a downloadable file and the reply links to it. If the prompt tells the model to
reproduce the document as its reply, it pastes the whole document into the chat instead.
"""

import unittest
from pathlib import Path

PROMPT = (Path(__file__).resolve().parent.parent / "resources" / "prompts" / "legal_prompt.txt").read_text(encoding="utf-8")


class LegalPromptDocumentDraftingTests(unittest.TestCase):
    def test_prompt_says_drafting_tools_are_available(self):
        self.assertIn("generate_legal_document", PROMPT)
        self.assertNotIn("Those tools are not available in this legal persona whitelist. If asked, explain you can help with jurisprudence/RA research in this mode", PROMPT)

    def test_prompt_does_not_tell_the_model_to_reproduce_the_document(self):
        self.assertNotIn("reproduce its `content`", PROMPT)
        self.assertNotIn("verbatim, as your reply", PROMPT)

    def test_a_users_own_case_title_is_not_treated_as_an_authority_to_look_up(self):
        # The general research rule says to search a named party case. For a drafting request the
        # title is the user's own proceeding: searching it made the reply say the case "could not be
        # retrieved", which is noise in a message that should just link the document.
        self.assertIn("is not an authority", PROMPT)
        self.assertIn("do not search for it", PROMPT)
        self.assertIn("never mention it in your reply as an authority", PROMPT)

    def test_prompt_tells_the_model_to_link_instead(self):
        self.assertIn("do NOT reproduce, quote or paste its text", PROMPT)
        self.assertIn("inline link", PROMPT)


UK_PROMPT = (Path(__file__).resolve().parent.parent / "resources" / "prompts" / "legal_prompt_uk.txt").read_text(encoding="utf-8")


class LegalPromptUkCitationAndQuoteWordingTests(unittest.TestCase):
    """Wording pins for ILL-ARB A3/B2/B3: these cannot prove the model obeys the rules, only that
    the prompt still states them (and no longer tells the model to repeat citations)."""

    def test_prompt_no_longer_tells_the_model_to_repeat_citations(self):
        self.assertNotIn("again at the end of the paragraphs they support", UK_PROMPT)
        self.assertIn("Never repeat an identical citation or link", UK_PROMPT)
        self.assertIn("never leave a proposition uncited", UK_PROMPT)

    def test_different_provision_of_a_cited_document_gets_its_own_pinpoint(self):
        self.assertIn("DIFFERENT provision or paragraph of a document you have already cited", UK_PROMPT)
        self.assertIn("pinpoint in both the link text and the sentence", UK_PROMPT)

    def test_exhibit_quote_marks_mean_exact_words(self):
        self.assertIn("Use them only for text copied exactly", UK_PROMPT)
        self.assertIn("paraphrase WITHOUT quote marks", UK_PROMPT)

    def test_remedies_questions_must_research_every_regime_and_its_cap(self):
        # ILL-ARB GT1: the answers to a combined unfair dismissal + Equality Act schedule-of-loss
        # question never said the compensatory award is capped and EqA compensation is not.
        self.assertIn("identify EVERY statutory regime", UK_PROMPT)
        self.assertIn("including any statutory cap, limit or deduction on that regime's award", UK_PROMPT)
        self.assertIn("state which award is capped and which is not", UK_PROMPT)
        self.assertIn("Do not state a cap, figure or limit you have not fetched", UK_PROMPT)

    def test_the_prompt_no_longer_demands_a_recap_or_repeated_authorities(self):
        # ILL-ARB A3: the prompt itself asked for a "direct-answer reprise", for authorities to be
        # named "again after the sentences they support", and for nested "Bottom line" labels.
        self.assertNotIn("direct-answer reprise → one clarifying question", UK_PROMPT)
        self.assertNotIn("and again after the sentences they support", UK_PROMPT)
        self.assertNotIn("Key point / Bottom line", UK_PROMPT)
        self.assertIn("No closing recap", UK_PROMPT)
        self.assertIn("Cite each authority once, where you first rely on it", UK_PROMPT)

    def test_each_issue_is_treated_in_one_place_without_losing_points(self):
        # Repetition comes out; content never does. A word budget was tried and rejected: cutting a
        # lengthy, non-repetitive answer to a number leaves the user with an incomplete solution.
        self.assertIn("No repetition, full completeness", UK_PROMPT)
        self.assertIn("never drop an issue, authority, gap, step or piece of draft wording to save space", UK_PROMPT)
        self.assertIn("Remove only repetition", UK_PROMPT)
        self.assertIn("if two headings would cover the same point", UK_PROMPT)
        self.assertNotIn("3,500 words", UK_PROMPT)
        self.assertNotIn("12 sections", UK_PROMPT)

    def test_examples_use_placeholders_not_invented_sums(self):
        # A live answer built tables around made-up sums ("Full loss of earnings: 40,000").
        self.assertIn("No invented figures", UK_PROMPT)
        self.assertIn("labelled placeholders (Amount A, Period 1)", UK_PROMPT)

    def test_proposed_wording_is_never_a_blockquote(self):
        # The gate deleted proposed schedule wording set as a blockquote, 14-15 notices per answer.
        self.assertIn("Proposed or draft wording", UK_PROMPT)
        self.assertIn("never as a blockquote", UK_PROMPT)

    def test_attribution_requires_the_text_to_say_it(self):
        self.assertIn("only if that text actually says it", UK_PROMPT)


if __name__ == "__main__":
    unittest.main()
