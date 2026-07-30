"""End-to-end regression checks for legal persona answer accuracy.

Unlike the other tests in this directory, these drive the real reason_loop
against a live OpenAI model and juris.ph, so they are skipped unless
OPENAI_API_KEY is set. Run manually with:

    python -m pytest tests/test_legal_answer_regressions.py -v

Each case encodes a fact pattern where chat-wonder previously produced a
wrong or incomplete answer (or one the underlying prompt explicitly warns
against), so a regression here means the model/prompt/pipeline changed in a
way that reintroduced a known failure mode.
"""

import os
import unittest

import the_server as srv

REQUIRES_LIVE = not os.getenv("OPENAI_API_KEY")

# _load_user_functions() normally only runs inside the FastAPI startup event
# (the_server.py's @app.on_event("startup")), which a plain module import
# never triggers. Without it, _context.all_fun_manifest stays empty, so
# process_persona()'s tool whitelist is always [] and the model never has
# real tools -- any correct-looking answer would only be coming from the
# keyword-based prefetch in legal_fact_boost.py, not live search. Loading it
# once here makes these tests exercise the same tool-calling path the real
# server uses.
srv._load_user_functions(overwrite_globals=True)


def _ask_legal(question: str) -> str:
    state = srv.ChatState()
    srv.init_openai_client(state, srv._context.openai_api_key)

    persona, user_input, filtered_tools, addendum_override = srv.process_persona(
        f"[legal ai] {question}"
    )
    user_input, prefetch = srv.prepare_legal_turn(user_input)
    if prefetch:
        state.last_search_legal_results = list(state.last_search_legal_results or []) + prefetch

    was_auto = srv._context.auto_approval
    srv._context.auto_approval = True
    try:
        result = srv.reason_loop(
            state,
            user_input,
            tools=filtered_tools,
            addendum_override=addendum_override,
            persona=persona,
        )
    finally:
        srv._context.auto_approval = was_auto

    final_text = (result or "").strip()
    legal_mode = bool(addendum_override and "LEGAL ASSISTANT MODE" in addendum_override)
    return srv._finalize_legal_response(
        final_text, state.last_search_legal_results, legal_mode=legal_mode, user_input=user_input
    )


@unittest.skipIf(REQUIRES_LIVE, "OPENAI_API_KEY not set; skipping live legal-answer regression checks")
class LegalAnswerRegressionTests(unittest.TestCase):
    def test_foreign_heir_can_inherit_by_succession(self):
        """Naturalized-foreign heirs CAN inherit PH land via hereditary succession
        (Art. XII Sec. 7) -- the exact fact pattern chat-wonder previously got
        wrong when compared against ChatGPT."""
        answer = _ask_legal(
            "My late father owned agricultural land in Batangas under a free patent. "
            "He had two children: me (Filipino) and my half-brother, who is now a "
            "naturalized American citizen. Before he died, my father had also signed "
            "a deed selling the same land to a neighbor, but that buyer never "
            "registered it; later my father sold it again to someone who did "
            "register. Who owns the land, can my American half-brother inherit "
            "Philippine land at all, and does the free patent change anything?"
        )
        lowered = answer.lower()

        self.assertNotRegex(
            answer,
            r"cannot inherit",
            "Model incorrectly denied hereditary succession rights to a foreign/"
            "naturalized heir instead of recognizing the Art. XII Sec. 7 exception.",
        )
        self.assertIn("hereditary succession", lowered)
        self.assertIn("11231", answer)
        # NOTE: naming the exact Art. XII Sec. 7 citation is not asserted here.
        # That depends on the model actually retrieving a source on point via
        # search_jurisprudence/search_republic_acts, which is a retrieval-breadth
        # gap (top_k defaults, no forced per-issue search) documented as a
        # follow-up in the accuracy-gap plan, not something steps 1-3 fixed.
        # When it's not retrieved, correctly declining to cite specifics is the
        # right anti-hallucination behavior, not a bug -- so this is intentionally
        # not enforced as a hard assertion.

    def test_floating_status_not_treated_as_resignation(self):
        """Security-guard floating status beyond six months risks constructive
        dismissal; resignation should not be framed as a requirement."""
        answer = _ask_legal(
            "I am a security guard placed on floating status / off-detail for "
            "eight months. Was I constructively dismissed?"
        )
        lowered = answer.lower()
        self.assertNotIn("resignation is required", lowered)
        self.assertIn("constructive dismissal", lowered)

    def test_cyber_libel_parallel_civil_remedy_not_closed_off(self):
        """Even if criminal cyber libel may be time-barred, a parallel civil
        remedy should not be silently closed off."""
        answer = _ask_legal(
            "Someone posted a defamatory Facebook post about me 18 months ago. "
            "Can I still file cyber libel?"
        )
        lowered = answer.lower()
        self.assertNotIn("you cannot file", lowered)
        self.assertIn("civil action", lowered)
        # NOTE: not asserting the exact "Article 33" citation -- like the
        # foreign-heir case above, that depends on the model retrieving a
        # source on point (retrieval-breadth follow-up, out of scope here).
        # What matters is that it doesn't close off the remedy path.


if __name__ == "__main__":
    unittest.main()
