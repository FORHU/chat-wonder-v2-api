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


if __name__ == "__main__":
    unittest.main()
