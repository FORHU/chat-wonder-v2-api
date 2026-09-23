"""Tests for #72 — draft_pleading (the free-form pleading path) and the fixes around it:
persona reachability (legal_whitelist), and the tool actually being called with the real
grounds/facts rather than dropping them. No live OpenAI call — the client is faked the same
way tests/test_generate_decision_records.py fakes it, since draft_pleading builds its own
OpenAI() client internally rather than taking one from ChatState.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)
import resources.functions.user_functions as uf  # noqa: E402  (loaded above)


def _completion_returning(content: str):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class _FakeClient:
    def __init__(self, content: str):
        self.content = content
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return _completion_returning(self.content)


DRAFTED_TEXT = (
    "REPUBLIC OF THE PHILIPPINES\nREGIONAL TRIAL COURT\nBranch 10, City of Makati\n\n"
    "...\n\nWHEREFORE, premises considered, it is respectfully prayed that the Motion for "
    "Reconsideration be DENIED for lack of merit."
)


class DraftPleadingTests(unittest.TestCase):
    def test_missing_required_fields_returns_error_without_calling_openai(self):
        fake = _FakeClient(DRAFTED_TEXT)
        with patch("openai.OpenAI", return_value=fake):
            r = uf.draft_pleading(pleading_type="", case_facts="", grounds="")
        self.assertFalse(r["success"])
        self.assertEqual(r["error"], "Missing required fields")
        self.assertEqual(set(r["missing_fields"]), {"pleading_type", "case_facts", "grounds"})
        self.assertEqual(fake.calls, [])  # never reaches the OpenAI call

    def test_successful_draft_returns_full_content_verbatim(self):
        fake = _FakeClient(DRAFTED_TEXT)
        with patch("openai.OpenAI", return_value=fake):
            r = uf.draft_pleading(
                pleading_type="Opposition to Motion for Reconsideration",
                case_facts="Plaintiff was served with the assailed Order on 1 June 2026.",
                grounds="The motion raises no new argument and merely rehashes points already resolved by the Order.",
                court="Regional Trial Court, Branch 10, City of Makati",
                case_title="Dela Cruz v. Santos",
                case_number="Civil Case No. 12345",
            )
        self.assertTrue(r["success"])
        self.assertEqual(r["pleading_type"], "Opposition to Motion for Reconsideration")
        # The full drafted text must come back unmodified — this is the exact bug #72
        # reports: sometimes the pipeline ends up with a description instead of the
        # document. At the draft_pleading level, that means content must be the model's
        # actual output, untouched.
        self.assertEqual(r["content"], DRAFTED_TEXT)
        self.assertIn("disclaimer", r)
        self.assertIn("next_steps", r)

    def test_grounds_and_facts_actually_reach_the_prompt(self):
        """Root-cause-1 regression: the old generate_legal_document silently rejected
        anything outside its 6-type enum. draft_pleading must not just accept a pleading
        request — it must put the caller's real facts/grounds into what it sends to the
        model, not a generic placeholder."""
        fake = _FakeClient(DRAFTED_TEXT)
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading(
                pleading_type="Motion to Dismiss",
                case_facts="UNIQUE_FACT_MARKER_998",
                grounds="UNIQUE_GROUNDS_MARKER_777",
            )
        self.assertEqual(len(fake.calls), 1)
        sent_messages = fake.calls[0]["messages"]
        user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
        system_content = next(m["content"] for m in sent_messages if m["role"] == "system")
        self.assertIn("UNIQUE_FACT_MARKER_998", user_content)
        self.assertIn("UNIQUE_GROUNDS_MARKER_777", user_content)
        # Root-cause-2 regression: the system prompt itself must forbid the
        # description-instead-of-document failure mode — the drafter is told to write the
        # actual document, not describe what it should contain.
        self.assertIn("Never respond with a description", system_content)

    def test_no_authorities_forbids_inventing_a_citation(self):
        """#73: with no verified research handed in, the drafter must not cite from memory."""
        fake = _FakeClient(DRAFTED_TEXT)
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading(pleading_type="Motion to Dismiss", case_facts="Facts.", grounds="Grounds.")
        user_content = next(m["content"] for m in fake.calls[0]["messages"] if m["role"] == "user")
        self.assertIn("Never invent a case name, citation, or statutory section from memory", user_content)

    def test_authorities_are_passed_verbatim_into_the_prompt(self):
        """#73: verified citations fetched earlier in the conversation must reach the drafting
        prompt exactly as given, with an instruction not to cite anything outside that list."""
        fake = _FakeClient(DRAFTED_TEXT)
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading(
                pleading_type="Opposition",
                case_facts="Facts.",
                grounds="Grounds.",
                authorities=[
                    {"citation": "Heirs of Malate v. Gamboa, G.R. No. 170338, Dec. 8, 2010", "proposition": "a mortgagee in good faith is protected"},
                ],
            )
        user_content = next(m["content"] for m in fake.calls[0]["messages"] if m["role"] == "user")
        self.assertIn("Heirs of Malate v. Gamboa, G.R. No. 170338, Dec. 8, 2010", user_content)
        self.assertIn("a mortgagee in good faith is protected", user_content)
        self.assertIn("Do not cite any case, statute, or provision that is not in that list", user_content)

    def test_manifest_declares_authorities_field_on_draft_pleading(self):
        tool = next(x["function"] for x in srv._context.all_fun_manifest if x["function"]["name"] == "draft_pleading")
        self.assertIn("authorities", tool["parameters"]["properties"])
        self.assertNotIn("authorities", tool["parameters"]["required"])

    def test_optional_fields_default_to_blanks_not_omitted(self):
        fake = _FakeClient(DRAFTED_TEXT)
        with patch("openai.OpenAI", return_value=fake):
            uf.draft_pleading(
                pleading_type="Manifestation",
                case_facts="Some facts.",
                grounds="Some grounds.",
            )
        sent_messages = fake.calls[0]["messages"]
        user_content = next(m["content"] for m in sent_messages if m["role"] == "user")
        self.assertIn("___", user_content)  # court/case_title/case_number left blank
        self.assertIn("infer the standard prayer", user_content)  # relief_sought default


class PersonaReachabilityTests(unittest.TestCase):
    """#72's fix is only real if the legal persona can actually see the tool —
    see #74's own root cause (a tool absent from a persona's whitelist is
    unreachable no matter what it does)."""

    def test_draft_pleading_is_in_the_legal_persona_toolset(self):
        _persona, _cleaned, filtered_tools, _addendum = srv.process_persona("[legal ai] draft an opposition")
        names = {t["function"]["name"] for t in (filtered_tools or [])}
        self.assertIn("draft_pleading", names)

    def test_draft_pleading_is_not_in_the_uk_legal_persona_toolset(self):
        # Deliberate per #79 — the UK sibling doesn't exist yet.
        _persona, _cleaned, filtered_tools, _addendum = srv.process_persona("[legal ai uk] draft an opposition")
        names = {t["function"]["name"] for t in (filtered_tools or [])}
        self.assertNotIn("draft_pleading", names)


if __name__ == "__main__":
    unittest.main()
