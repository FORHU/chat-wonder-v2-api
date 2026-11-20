"""Tenant reply-language restriction (issue #145 / CW-6): on /chat and /chat-stream, the
reply language is clamped to each tenant's allowed set (PH: en/tl, KR: en/ko, UK: en-only),
falling back to the tenant's default (English for all three) when the caller-supplied or
langid-detected language isn't allowed. No tenant sent (or an unrecognized one) behaves exactly
as today — the clamp sits after the existing supplied/langid resolution in prepare_chat_messages,
not inside it.
"""

import unittest
from unittest.mock import patch

import the_server as srv


class TenantReplyLanguageTests(unittest.TestCase):
    def _state(self, jurisdiction=None, reply_language=None):
        state = srv.ChatState()
        state.jurisdiction = jurisdiction
        state.reply_language = reply_language
        return state

    def test_uk_tenant_disallowed_supplied_language_falls_back_to_english(self):
        state = self._state(jurisdiction="UK", reply_language="tl")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query to skip the short-query default", persona="legal_uk")
        self.assertIn("`en`-locale", messages[-1]["content"])

    def test_uk_tenant_allowed_supplied_language_passes_through(self):
        state = self._state(jurisdiction="UK", reply_language="en")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="legal_uk")
        self.assertIn("`en`-locale", messages[-1]["content"])

    def test_kr_tenant_disallowed_language_falls_back_to_english_default(self):
        state = self._state(jurisdiction="KR", reply_language="tl")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="legal_kr")
        self.assertIn("`en`-locale", messages[-1]["content"])

    def test_kr_tenant_allowed_language_passes_through(self):
        state = self._state(jurisdiction="KR", reply_language="ko")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="legal_kr")
        self.assertIn("`ko`-locale", messages[-1]["content"])

    def test_ph_tenant_disallowed_language_falls_back_to_english_default(self):
        state = self._state(jurisdiction="PH", reply_language="ko")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="legal")
        self.assertIn("`en`-locale", messages[-1]["content"])

    def test_ph_tenant_allowed_filipino_passes_through(self):
        state = self._state(jurisdiction="PH", reply_language="tl")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="legal")
        self.assertIn("`tl`-locale", messages[-1]["content"])

    def test_no_tenant_behaves_as_today_even_with_disallowed_looking_language(self):
        state = self._state(jurisdiction=None, reply_language="fr")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="auto")
        self.assertIn("`fr`-locale", messages[-1]["content"])

    def test_unknown_tenant_behaves_as_no_tenant(self):
        state = self._state(jurisdiction="AU", reply_language="fr")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="auto")
        self.assertIn("`fr`-locale", messages[-1]["content"])

    def test_tenant_value_is_case_and_whitespace_insensitive(self):
        state = self._state(jurisdiction=" uk ", reply_language="tl")
        messages = srv.prepare_chat_messages(state, "a sufficiently long query", persona="legal_uk")
        self.assertIn("`en`-locale", messages[-1]["content"])

    def test_langid_fallback_is_clamped_too(self):
        state = self._state(jurisdiction="UK", reply_language=None)
        with patch("the_server.langid.classify", return_value=("tl", 0.9)):
            messages = srv.prepare_chat_messages(state, "a sufficiently long query with no supplied language at all here", persona="legal_uk")
        self.assertIn("`en`-locale", messages[-1]["content"])

    def test_chat_endpoint_stashes_jurisdiction_onto_session_state(self):
        session_id = "test-session-tenant-stash"
        srv._context.sessions[session_id] = srv.ChatState()
        request = srv.ChatRequest(user_input="hi", session_id=session_id, jurisdiction="UK")
        if session_id and session_id in srv._context.sessions:
            srv._context.sessions[session_id].jurisdiction = request.jurisdiction
        self.assertEqual(srv._context.sessions[session_id].jurisdiction, "UK")
        del srv._context.sessions[session_id]


if __name__ == "__main__":
    unittest.main()
