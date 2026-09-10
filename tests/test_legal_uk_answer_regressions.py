"""End-to-end regression checks for the legal_uk persona's answer accuracy.

Mirrors tests/test_legal_answer_regressions.py's approach (drives the real
reason_loop against a live OpenAI model and the UK Legal MCP), but for
legal_uk instead of legal (PH). Skipped unless OPENAI_API_KEY is set. Run
manually with:

    python -m pytest tests/test_legal_uk_answer_regressions.py -v

Unlike the PH suite, this persona had NO live regression coverage before this
file: legal_fact_boost.prepare_legal_turn() (general protocol injection +
doctrine hints + prefetch) is only wired up for persona == "legal" in
the_server.py, never for "legal_uk" -- so legal_uk relies entirely on
resources/prompts/legal_prompt_uk.txt being followed correctly, with no
automated check that it actually is. Each case below targets a rule that
prompt states explicitly (often flagged there as a known/recurring failure),
so a regression here means the model/prompt/pipeline stopped honoring it.
"""

import os
import re
import unittest
import uuid

import the_server as srv

REQUIRES_LIVE = not os.getenv("OPENAI_API_KEY")

# See test_legal_answer_regressions.py for why this must run once at import time.
srv._load_user_functions(overwrite_globals=True)

_CITATION_LINK = re.compile(
    r'<a href="([^"]+)" class="legal-ref (?:law|jurisprudence)"'
    r'|\[[^\]]+ (?:Law|Jurisprudence)\]\(([^)]+)\)'
)
_ALLOWED_CITATION_DOMAINS = ("https://www.legislation.gov.uk/", "https://caselaw.nationalarchives.gov.uk/")


def _ask_legal_uk(question: str) -> str:
    state = srv.ChatState()
    srv.init_openai_client(state, srv._context.openai_api_key)

    # execute_function_call() only accumulates tool results into
    # state.last_search_legal_results when it can look the SAME state object
    # up via _context.sessions[session_id] (the_server.py:1159-1160,
    # :1251/:1312) -- exactly what the real /chat endpoint sets up via
    # GET /session-id before any turn runs. Without registering one here, the
    # citation pool stays empty for the whole run and gate_unverified_legal_urls/
    # strip_unverified_blockquotes would then treat every link and quote as
    # unverified regardless of whether it was actually correct.
    session_id = str(uuid.uuid4())
    srv._context.sessions[session_id] = state

    # No prepare_legal_turn() here -- the real /chat endpoint never calls it for
    # legal_uk (the_server.py:2463 gates that on persona == "legal" only), so
    # this harness must match that to actually test what production runs.
    persona, user_input, filtered_tools, addendum_override = srv.process_persona(
        f"[legal ai uk] {question}"
    )

    try:
        result = srv.reason_loop(
            state,
            user_input,
            session_id=session_id,
            tools=filtered_tools,
            addendum_override=addendum_override,
            persona=persona,
        )
    finally:
        srv._context.sessions.pop(session_id, None)

    final_text = (result or "").strip()
    legal_mode = bool(addendum_override and "LEGAL ASSISTANT MODE" in addendum_override)
    return srv._finalize_legal_response(
        final_text, state.last_search_legal_results, legal_mode=legal_mode, user_input=user_input
    )


@unittest.skipIf(REQUIRES_LIVE, "OPENAI_API_KEY not set; skipping live legal_uk-answer regression checks")
class LegalUkAnswerRegressionTests(unittest.TestCase):
    def test_extent_not_assumed_uk_wide(self):
        """legal_prompt_uk.txt calls omitting the `extent` check 'a recurring,
        serious error': GB employment statutes like ERA 1996 do not
        automatically extend to Northern Ireland, which has its own mirror
        legislation (e.g. the Employment Rights (Northern Ireland) Order
        1996). The model must not recite ERA 1996 as if it protects an NI
        worker without flagging that distinction."""
        answer = _ask_legal_uk(
            "I work in Belfast, Northern Ireland. I was dismissed two days after "
            "reporting my employer to the police for tax fraud. Does the "
            "Employment Rights Act 1996 protect me as a whistleblower?"
        )
        lowered = answer.lower()
        self.assertIn("northern ireland", lowered)
        self.assertTrue(
            any(
                phrase in lowered
                for phrase in (
                    "does not extend",
                    "does not apply",
                    "separate legislation",
                    "different legislation",
                    "northern ireland order",
                    "mirror",
                    "equivalent legislation",
                    "own legislation",
                )
            ),
            "Answer should flag that GB employment legislation like ERA 1996 has "
            "different territorial extent for Northern Ireland, not recite it as "
            "if it applies uk-wide unqualified.",
        )

    def test_citation_links_use_verified_domains_only(self):
        """Citation Workflow (MANDATORY) forbids hand-built citation strings/URLs --
        every ' Law'/' Jurisprudence' link must come from citations_resolve's
        resolved_url, which only ever points at legislation.gov.uk or
        caselaw.nationalarchives.gov.uk. A link to any other domain means the
        model hand-guessed a URL instead of following the required chain."""
        answer = _ask_legal_uk(
            "What does the Companies Act 2006 say about a director's duty to "
            "avoid conflicts of interest?"
        )
        links = [html_href or md_href for html_href, md_href in _CITATION_LINK.findall(answer)]
        self.assertTrue(links, "Expected at least one ' Law'/' Jurisprudence' citation link in the answer.")
        for href in links:
            self.assertTrue(
                href.startswith(_ALLOWED_CITATION_DOMAINS),
                f"Citation href {href!r} is not a legislation.gov.uk / caselaw.nationalarchives.gov.uk "
                "URL -- looks hand-built rather than from citations_resolve.",
            )

    def test_multi_forum_fact_pattern_not_collapsed_to_one_remedy(self):
        """Dynamic Analysis rule (d): a fact pattern spanning more than one area
        of law must name each forum separately, not silently resolve to a
        single remedy or close off paths."""
        answer = _ask_legal_uk(
            "I live in tied accommodation provided by my employer in England. "
            "I was just dismissed after raising a health and safety complaint, "
            "and now my employer says I have 7 days to leave the property. "
            "What can I do?"
        )
        lowered = answer.lower()
        self.assertNotIn("you have no remedy", lowered)
        self.assertIn("tribunal", lowered)
        self.assertTrue(
            "possession" in lowered or "housing" in lowered or "county court" in lowered,
            "Answer should separately address the tied-accommodation/possession "
            "forum, not only the employment tribunal claim.",
        )


if __name__ == "__main__":
    unittest.main()
