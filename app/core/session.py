# -*- coding: utf-8 -*-
"""Chat session state and its on-disk persistence.

Extracted from the_server.py (chat-wonder-v2-api#182). ChatState itself has
no dependency on the server's global ApiContext; the persistence/lookup
helpers below take the live `sessions` dict as an explicit argument instead
of reaching for a module-level context, so this module stays usable without
the_server's other global state.
"""

from __future__ import annotations

import logging
import os
import pickle
import time
from typing import Optional

from fastapi import HTTPException

from llm_usage import make_openai_client

SESSION_TTL_SECONDS = 3600

# Session Persistence
SESSION_STORAGE_PATH = "sessions.pkl"
SESSION_RESTORE_WINDOW = 600  # only restore sessions used within the last 10 minutes


class ChatState:
    def __init__(self):
        self.prompt: list = []
        self.generated: list = []
        self.lookup: tuple = ([], [])
        self.summary: str = ""
        self.source_metadata: list = []
        self.last_search_legal_results: list = []
        self.last_turn_tool_log: list = []
        self.last_garment_result: dict = {}
        self.last_cosmetics_result: dict = {}
        self.last_maps_result: list = []
        self.last_nav_result: dict = {}
        self.last_tailor_result: dict = {}
        self.last_generated_file_result: dict = {}
        self.last_outfit_ids_result: list = []
        self.last_cosmetics_ids_result: list = []
        # Intake's find_lawyer/mark_case_ready — reset every turn (see process_persona's caller),
        # unlike last_outfit_ids_result above, so a result from turn N never resends on turn N+1.
        self.last_find_lawyer_result: list = []
        self.last_mark_case_ready: bool = False
        self.active_case_documents: list = []
        self.case_document_cache: dict = {}
        # None = not yet synced this process; empty set = this turn has no case docs.
        self.allowed_case_document_ids = None
        # This turn's raw user_input, captured before persona tag injection (weather/sitemap/
        # skin-analysis annotations aren't part of the actual question) — read by
        # execute_function_call to give get_case_document something to rank chunks against.
        self.current_query: str = ""
        # Every attached document's id/name/category, whether or not its chunks made this
        # turn's cut — see docs/adr/0005. Set by sync_active_case_documents.
        self.case_document_manifest: list = []
        # The case's Damages & Remedies model from ilovelawyer-api (request.case_damages), replaced
        # on every legal turn — None when the case has no heads or the turn isn't case-linked. See
        # _build_case_damages_injection.
        self.case_damages: Optional[dict] = None
        # Short per-session labels (F1, F2, ...) the model is shown instead of raw case-document
        # UUIDs in any prompt, so it has nothing UUID-shaped to copy into an answer or decision
        # record. Assigned once per document id and kept stable for the session's lifetime — see
        # sync_active_case_documents. doc_id_by_handle is the reverse map used to resolve a
        # handle the model sends back via execute_function_call.
        self.doc_handle_by_id: dict = {}
        self.doc_id_by_handle: dict = {}
        self.document_fetch_error: Optional[dict] = None
        self.confirmed_gender: str = ""
        self.sitemap_context: list = []
        self.openai_client = None
        self.last_used: float = time.time()
        self.user_id: Optional[str] = None
        # Set from ilovelawyer-api's intake WS payload, mirrors user_id — used by create_case's
        # session-scoped arg injection in execute_function_call.
        self.case_id: Optional[str] = None
        # Set at the start of every turn from ChatRequest.trace_turn_id (see broadcast_trace).
        # Read via getattr elsewhere: sessions restored from sessions.pkl predate this field.
        self.trace_turn_id: Optional[str] = None
        # HITL pending state
        self.pending_function_call: Optional[dict] = None
        self.pending_messages: Optional[list] = None
        self.pending_session_id: Optional[str] = None
        self.pending_tools: Optional[list] = None
        self.pending_addendum: Optional[str] = None
        # confidence.py telemetry: this turn's request opt-in (ChatRequest.confidence_live) and
        # the (token_text, h) pairs captured during streaming, built from logprobs.content[],
        # never from the yielded delta text — see confidence.group_words' docstring. Reset at the
        # start of every turn in streaming_run_function_chain; read via getattr elsewhere
        # (sessions restored from sessions.pkl predate these fields).
        self.confidence_live: bool = False
        self.confidence_tokens: Optional[list] = None

    def __getstate__(self):
        """Exclude the live OpenAI client from pickling — it wraps open sockets/threads and
        isn't serializable. Nothing is lost: every request path re-runs init_openai_client()
        unconditionally before touching state.openai_client (see /chat and the websocket
        handler), so a restored session just gets a fresh client on its next turn.

        confidence_tokens is per-turn telemetry with no reason to survive a restart (and no
        bound on how large a long session's accumulated data could grow), so it is stripped
        the same way."""
        state = self.__dict__.copy()
        state["openai_client"] = None
        state["confidence_tokens"] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)


def save_sessions_to_disk(sessions: dict) -> None:
    try:
        with open(SESSION_STORAGE_PATH, "wb") as f:
            pickle.dump(sessions, f)
        logging.debug(f"Saved {len(sessions)} sessions to disk.")
    except Exception as e:
        logging.error(f"Failed to save sessions: {e}")


def load_sessions_from_disk(sessions: dict) -> None:
    if not os.path.exists(SESSION_STORAGE_PATH):
        return
    try:
        with open(SESSION_STORAGE_PATH, "rb") as f:
            saved_sessions = pickle.load(f)
        now = time.time()
        restored = 0
        for sid, state in saved_sessions.items():
            if now - state.last_used < SESSION_RESTORE_WINDOW:
                sessions[sid] = state
                restored += 1
        logging.info(f"Restored {restored} active session(s) from disk.")
    except Exception as e:
        logging.error(f"Failed to load sessions: {e}")


def cleanup_sessions(sessions: dict) -> None:
    while True:
        now = time.time()
        expired = [sid for sid, s in list(sessions.items()) if now - s.last_used > SESSION_TTL_SECONDS]
        for sid in expired:
            del sessions[sid]
        if sessions:
            save_sessions_to_disk(sessions)
        time.sleep(600)


def get_session_state(sessions: dict, session_id):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return sessions[session_id]


def init_openai_client(state, api_key, base_url=None):
    params = {}
    if base_url:
        params["base_url"] = base_url
    state.openai_client = make_openai_client(api_key=api_key, **params)
