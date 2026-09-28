"""The tool-free `[extract]` persona used by ilovelawyer-api's DamagesExtractSvc.

The wiring tests run offline. The contract test drives the real reason_loop against a live
OpenAI model, so it is skipped unless OPENAI_API_KEY is set — it guards the output ilovelawyer-api's
damages-extract-parse.ts depends on: a parseable [DAMAGES] array whose quotes are copied from the
documents, with inputs rather than totals. Run manually with:

    python -m pytest tests/test_extract_persona.py -v
"""

import json
import os
import re
import unittest

import the_server as srv

REQUIRES_LIVE = not os.getenv("OPENAI_API_KEY")

# Same reason as test_legal_answer_regressions.py: without this the manifest stays empty, so a
# whitelist would be [] for every persona and "extract has no tools" would pass vacuously.
srv._load_user_functions(overwrite_globals=True)

PAYSLIP = "PAYSLIP - August 2025\nEmployee: Juan Dela Cruz\nBasic monthly salary: P27,000.00\nNet pay: P24,310.00"
COMPLAINT = (
    "WHEREFORE, complainant prays that respondent be ordered to pay P200,000.00 as moral damages, "
    "P100,000.00 as exemplary damages, and attorney's fees equivalent to ten percent (10%) of the total monetary award."
)

# A condensed copy of ilovelawyer-api's damages-extract prompt (src/constants/damages-extract.constants.ts):
# same output contract, fewer words.
PROMPT = f"""You are building a damages model for a litigation team in the Philippines. Case: Cruz v. Acme (illegal dismissal).
Categories: ACTUAL, MORAL, EXEMPLARY, ATTORNEYS_FEES, OTHER.

ALREADY IN THE DAMAGES MODEL (do not repeat):
(none)

DOCUMENTS:
--- DOCUMENT id: doc-pay | name: Payslip.pdf ---
{PAYSLIP}

--- DOCUMENT id: doc-cmp | name: Complaint.pdf ---
{COMPLAINT}

INSTRUCTIONS:
Use ONLY the document text above. Find each head of damages or monetary relief this case could claim where a document states a figure to build it from:
- a figure a pleading, demand letter or prayer already asks for;
- a wage, salary or other rate that a claim is computed from (e.g. a monthly salary in a payslip, for backwages under ACTUAL);
- a percentage asked for (e.g. attorney's fees as a percentage of the award).
Give each head's inputs, never a total.
"basis" is one of {{"kind":"FIXED","amount":<number>}}, {{"kind":"RATE_X_PERIOD","monthlyRate":<number>}} or {{"kind":"PERCENT_OF","percent":<number>,"categories":[...]}}.
Every number in "basis" must appear in "quote". "quote" is copied character-for-character from the document "documentId" (10-300 characters).

Respond with the machine-readable block below and nothing else, exactly in this format:
[DAMAGES]
[{{"category":"ACTUAL","label":"...","basis":{{"kind":"RATE_X_PERIOD","monthlyRate":0}},"legalBasis":null,"pendingEvidence":null,"documentId":"<id from above>","quote":"..."}}]
[/DAMAGES]"""


class ExtractPersonaWiringTests(unittest.TestCase):
    def test_extract_tag_selects_a_tool_free_persona(self):
        persona, cleaned, tools, addendum = srv.process_persona("[extract] Find the damages.")
        self.assertEqual(persona, "extract")
        self.assertEqual(cleaned, "Find the damages.")
        self.assertEqual(tools, [])
        self.assertIn("EXTRACTION MODE", addendum)
        self.assertIn("nothing else", addendum)

    def test_tag_is_case_insensitive_and_ignores_jurisdiction(self):
        persona, _cleaned, tools, _addendum = srv.process_persona("[EXTRACT] Find the damages.", "UK")
        self.assertEqual(persona, "extract")
        self.assertEqual(tools, [])

    def test_legal_tags_still_route_to_the_legal_personas(self):
        self.assertEqual(srv.process_persona("[legal ai] What is backwages?")[0], "legal")
        self.assertEqual(srv.process_persona("[legal ai uk] What is a basic award?")[0], "legal_uk")

    def test_a_longer_tag_that_merely_starts_with_extract_is_not_captured(self):
        self.assertNotEqual(srv.process_persona("[extractor] hello")[0], "extract")


@unittest.skipIf(REQUIRES_LIVE, "OPENAI_API_KEY not set; skipping live [DAMAGES] contract check")
class ExtractDamagesContractTests(unittest.TestCase):
    def test_reply_is_a_parseable_block_of_quoted_inputs(self):
        state = srv.ChatState()
        srv.init_openai_client(state, srv._context.openai_api_key)
        persona, user_input, tools, addendum = srv.process_persona(f"[extract] {PROMPT}")
        reply = (srv.reason_loop(state, user_input, tools=tools, addendum_override=addendum, persona=persona) or "").strip()

        match = re.search(r"\[DAMAGES\]([\s\S]*?)\[/DAMAGES\]", reply)
        self.assertIsNotNone(match, reply)
        body = re.sub(r"^```(?:json)?\s*|```$", "", match.group(1).strip())
        heads = json.loads(body)
        self.assertIsInstance(heads, list)
        self.assertGreater(len(heads), 0, reply)

        docs = {"doc-pay": PAYSLIP, "doc-cmp": COMPLAINT}
        squash = lambda t: re.sub(r"\s+", " ", t).strip().lower()  # noqa: E731
        for head in heads:
            self.assertIn(head.get("documentId"), docs, head)
            self.assertIn(squash(head["quote"]), squash(docs[head["documentId"]]), head)
            self.assertNotIn("total", {k.lower() for k in head}, head)
            self.assertNotIn("total", {k.lower() for k in (head.get("basis") or {})}, head)
        self.assertIn("ACTUAL", {h["category"] for h in heads}, reply)
