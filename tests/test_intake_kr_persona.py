"""Korean Case Intake (intake_kr): the KR site's version of the Case Intake persona.

Same posture and downstream handling as "intake" (tests/test_intake_persona.py) — gathers facts
for a lawyer, may explain general rights under Korean law with fetched citations, never assesses
the client's own case — with the Korean Law MCP's fetch-and-search tools instead of the UK ones,
and its own prompt. ilovelawyer-api sends the plain "[intake]" tag with jurisdiction "KR", the
same way it reaches legal_kr. Wiring-only tests: no OpenAI call, no live data.
"""

import os
import re
import unittest
from unittest.mock import patch

import the_server as srv

srv._load_user_functions(overwrite_globals=True)

# ilovelawyer-api's default KR case areas (src/constants/case-area.constants.ts, KR list).
KR_CASE_AREA_CODES = (
    "EMPLOYMENT",
    "FAMILY",
    "PROPERTY",
    "CRIMINAL",
    "BUSINESS",
    "PUBLIC_BODIES",
    "INTELLECTUAL_PROPERTY",
    "TAX",
    "MEDICAL",
    "OTHER",
)


class IntakeKrRoutingTests(unittest.TestCase):
    def test_intake_tag_with_kr_jurisdiction_resolves_to_intake_kr(self):
        persona, cleaned, tools, addendum = srv.process_persona("[intake] 사장님이 월급을 안 줘요", "KR")
        self.assertEqual(persona, "intake_kr")
        self.assertEqual(cleaned, "사장님이 월급을 안 줘요")
        self.assertTrue(addendum)
        self.assertEqual(
            {t["function"]["name"] for t in tools},
            {"search_law", "get_law_text", "search_decisions", "get_decision_text", "mark_case_ready", "create_case"},
        )

    def test_explicit_intake_kr_tag_resolves_without_a_jurisdiction(self):
        persona, cleaned, _tools, _addendum = srv.process_persona("[intake kr] 보증금을 못 받았어요")
        self.assertEqual(persona, "intake_kr")
        self.assertEqual(cleaned, "보증금을 못 받았어요")

    def test_jurisdiction_match_ignores_case_and_whitespace(self):
        persona, _c, _t, _a = srv.process_persona("[intake] hello", " kr ")
        self.assertEqual(persona, "intake_kr")

    def test_uk_intake_is_unchanged_without_kr(self):
        for jurisdiction in (None, "UK", "PH"):
            persona, _c, tools, addendum = srv.process_persona("[intake] my landlord won't fix the boiler", jurisdiction)
            self.assertEqual(persona, "intake", jurisdiction)
            self.assertIn("case_law_search", {t["function"]["name"] for t in tools})
            self.assertNotIn("KOREA", addendum)

    def test_legal_kr_is_unchanged(self):
        persona, _c, _t, _a = srv.process_persona("[legal ai] 해고 관련 질문", "KR")
        self.assertEqual(persona, "legal_kr")

    def test_never_offers_uk_research_applied_analysis_find_lawyer_or_case_documents(self):
        _p, _c, tools, _a = srv.process_persona("[intake] hello", "KR")
        names = {t["function"]["name"] for t in tools}
        # UK sources mean nothing for a Korean case.
        for tool in ("case_law_search", "judgment_get_header", "legislation_search", "legislation_get_section"):
            self.assertNotIn(tool, names)
        # legal_kr's applied-analysis tools: intake informs, it never builds an argument.
        for tool in ("legal_research", "legal_analysis", "discover_tools_kr", "execute_tool_kr"):
            self.assertNotIn(tool, names)
        # KR lawyers don't take client requests yet, so find_lawyer could only return nobody.
        self.assertNotIn("find_lawyer", names)
        # Same reason as the UK intake: no scope check without sync_active_case_documents.
        self.assertNotIn("get_case_document", names)


class IntakeKrPromptTests(unittest.TestCase):
    def setUp(self):
        _p, _c, _t, self.addendum = srv.process_persona("[intake] hello", "KR")
        self.flat = " ".join(self.addendum.split())

    def test_is_detected_as_an_intake_addendum_and_never_enters_legal_mode(self):
        # The HITL-resume paths only have the addendum text, not the persona id.
        self.assertTrue(srv._is_intake_addendum(self.addendum))
        self.assertNotIn("LEGAL ASSISTANT MODE", self.addendum)

    def test_forbids_applying_law_to_the_users_facts(self):
        self.assertIn("conclude whether THIS user has a case", self.flat)

    def test_cites_korean_law_by_name_never_as_a_link(self):
        self.assertIn("BY NAME, never as a link", self.flat)
        self.assertIn("never write a URL or a markdown link", self.flat)

    def test_never_states_a_deadline_or_amount_it_did_not_fetch(self):
        self.assertIn("never stated from memory", self.flat)

    def test_lists_exactly_the_kr_case_area_codes(self):
        listed = set(re.findall(r"^  ([A-Z_]+) \(", self.addendum, flags=re.MULTILINE))
        self.assertEqual(listed, set(KR_CASE_AREA_CODES))

    def test_never_sends_uk_only_fields(self):
        self.assertIn("NEVER send `uk_jurisdiction` or `outside_uk`", self.flat)
        self.assertIn('Ignore any "Part of the UK" line', self.flat)

    def test_says_lawyer_matching_is_not_open_in_korea(self):
        self.assertIn("isn't available in Korea yet", self.flat)
        self.assertIn("Never name, recommend or describe a specific lawyer", self.flat)

    def test_replies_in_the_users_language(self):
        self.assertIn("Reply in the language the user writes in", self.flat)
        self.assertIn('no "redundancy"', self.flat)

    def test_keeps_the_uk_intakes_interviewing_rules(self):
        for rule in (
            "the single worst thing this mode can do",
            "Never ask about anything on that list",
            "call `mark_case_ready()`",
            "never more than once within the same reply",
            "only ever narrate a successful write",
        ):
            self.assertIn(rule, self.flat)


class IntakeKrDownstreamTests(unittest.TestCase):
    def test_uses_the_intake_model_override(self):
        with patch.dict(os.environ, {"LEGAL_USE_RESPONSES_API": "true", "INTAKE_REASONING_EFFORT": ""}):
            self.assertEqual(srv._intake_model_override("intake_kr"), srv._intake_model_override("intake"))
            self.assertIsNotNone(srv._intake_model_override("intake_kr")[0])

    def test_case_context_gets_the_intake_label(self):
        self.assertEqual(
            srv._document_context_injection("intake_kr", "ctx"),
            srv._document_context_injection("intake", "ctx"),
        )

    def test_every_persona_label_dict_knows_intake_kr(self):
        source = open("the_server.py", encoding="utf-8").read()
        label_dicts = re.findall(r"_persona_label = \{[^}]*\}", source)
        self.assertGreaterEqual(len(label_dicts), 2)
        for d in label_dicts:
            self.assertIn('"intake_kr"', d)

    def test_no_code_path_still_checks_for_the_uk_intake_alone(self):
        # Every intake-specific branch must cover intake_kr too (INTAKE_PERSONAS) — a missed one
        # would, e.g., stream an un-gated answer live or drop the [CASE_READY] frame.
        source = open("the_server.py", encoding="utf-8").read()
        code_lines = [line for line in source.splitlines() if not line.lstrip().startswith("#")]
        offenders = [line.strip() for line in code_lines if re.search(r'persona\s*[!=]=\s*"intake"|"intake"\)', line)]
        self.assertEqual(offenders, [])


class IntakeKrLanguageTests(unittest.TestCase):
    def _language_line(self, persona, query, state=None):
        messages = srv.prepare_chat_messages(state or srv.ChatState(), query, addendum_override="x", persona=persona)
        return re.search(r"respond in the `([a-z]+)`-locale", messages[1]["content"]).group(1)

    def _conversation(self, client_said, ai_said):
        state = srv.ChatState()
        state.prompt = [client_said]
        state.generated = [ai_said]
        return state

    def test_a_korean_message_is_answered_in_korean(self):
        # Three words: the short-query default would otherwise force English.
        self.assertEqual(self._language_line("intake_kr", "월급을 못 받았어요"), "ko")

    def test_an_english_message_is_answered_in_english(self):
        self.assertEqual(self._language_line("intake_kr", "I lost my job"), "en")

    def test_a_short_latin_reply_stays_korean_in_a_korean_conversation(self):
        state = self._conversation("사장님이 월급을 안 줘요", "많이 힘드셨겠어요. 어느 지역에서 있었던 일인가요?")
        self.assertEqual(self._language_line("intake_kr", "ok", state), "ko")

    def test_a_short_latin_reply_stays_english_in_an_english_conversation(self):
        state = self._conversation("My boss hasn't paid my wages for two months", "I'm sorry to hear that. Where in Korea did this happen?")
        self.assertEqual(self._language_line("intake_kr", "ok", state), "en")

    def test_other_personas_keep_the_short_query_default(self):
        self.assertEqual(self._language_line("intake", "월급을 못 받았어요"), "en")

    def test_a_caller_supplied_language_still_wins(self):
        state = srv.ChatState()
        state.reply_language = "en"
        self.assertEqual(self._language_line("intake_kr", "월급을 못 받았어요", state), "en")


if __name__ == "__main__":
    unittest.main()
