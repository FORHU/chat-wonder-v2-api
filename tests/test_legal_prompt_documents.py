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

    def test_prompt_tells_the_model_to_link_instead(self):
        self.assertIn("do NOT reproduce, quote or paste its text", PROMPT)
        self.assertIn("inline link", PROMPT)


if __name__ == "__main__":
    unittest.main()
