# -*- coding: utf-8 -*-

import os
import asyncio
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import uuid
import json
import time
import re
import io
import sys
import subprocess
import importlib.util
import logging
import pickle
import zipfile
import tempfile
import shutil
from threading import Thread
from typing import Optional, List

import dotenv
import pandas as pd
import tiktoken
import langid
from openai import OpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from legal_rag.router import (
    LegalAskRequest as RouterLegalAskRequest,
    legal_ask as legal_rag_ask,
    legal_search as legal_rag_search,
    get_legal_document as legal_rag_get_document,
    router as legal_rag_router,
)
import s3_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# API Context (global state)
# ---------------------------------------------------------------------------

class ApiContext:
    embeddings = None
    addendum = ""
    fun_manifest: list = []
    fun_names: list = []
    user_functions: dict = {}
    sessions: dict = {}
    db_pool = None
    auto_approval: bool = False  # HITL: when True all tool calls execute without asking

_context = ApiContext()
_context.FUNCTIONS_DIR = os.path.join("resources", "functions")
_context.char_encodings = ["utf-8", "cp949", "euc-kr", "latin1"]
_context.__END__ = "__END__"

os.makedirs(_context.FUNCTIONS_DIR, exist_ok=True)

dotenv.load_dotenv()
user_env_path = os.path.join("resources", "functions", "user_functions.env")
if os.path.exists(user_env_path):
    dotenv.load_dotenv(user_env_path, override=True)

_context.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
_context.embedding_model: str = os.getenv("LEGAL_EMBEDDING_MODEL", "text-embedding-3-small")
_context.model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
_context.temperature: float = 1.0
_context.informed: bool = True
_context.show_clues: list = []
_context.expertise: str = "General"
_context.tone: str = "factual"

if os.path.exists("addendum.txt"):
    try:
        with open("addendum.txt", "r", encoding="utf-8") as f:
            _context.addendum = f.read().strip()
    except Exception as e:
        logging.warning(f"addendum.txt load failed: {e}")

# ---------------------------------------------------------------------------
# Chat Session State
# ---------------------------------------------------------------------------

class ChatState:
    def __init__(self):
        self.prompt: list = []
        self.generated: list = []
        self.lookup: tuple = ([], [])
        self.summary: str = ""
        self.source_metadata: list = []
        self.last_search_legal_results: list = []
        self.openai_client = None
        self.last_used: float = time.time()
        # HITL pending state
        self.pending_function_call: Optional[dict] = None
        self.pending_messages: Optional[list] = None
        self.pending_session_id: Optional[str] = None
        self.pending_tools: Optional[list] = None
        self.pending_addendum: Optional[str] = None

SESSION_TTL_SECONDS = 3600

def cleanup_sessions():
    while True:
        now = time.time()
        expired = [sid for sid, s in list(_context.sessions.items()) if now - s.last_used > SESSION_TTL_SECONDS]
        for sid in expired:
            del _context.sessions[sid]
        time.sleep(600)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------

app = FastAPI(title="Chat Wonder v2 API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(legal_rag_router)

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_input: str = ""
    user_history_select: Optional[str] = None
    session_id: Optional[str] = None
    document_context: Optional[str] = None

class ApproveRequest(BaseModel):
    session_id: str
    decision: str = "approved"  # approved | rejected | skipped_continue
    comments: Optional[str] = None

class SetHitlRequest(BaseModel):
    auto_approval: bool = False

class ExportRequest(BaseModel):
    file_type: str = ""
    file_name: str = ""
    session_id: str = None

class ImportRequest(BaseModel):
    conversation: str
    session_id: str = None

class EmotionRequest(BaseModel):
    text: str
    session_id: str = None

class DocumentUploadUrlRequest(BaseModel):
    filename: str
    content_type: str

class AnalyzeS3DocumentRequest(BaseModel):
    s3_key: str
    filename: Optional[str] = None
    session_id: Optional[str] = None

class SynthesizeDocumentsRequest(BaseModel):
    summaries: List[str]

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    logging.info("Starting Chat Wonder v2...")
    try:
        from psycopg2 import pool as pg_pool
        db_url = os.getenv("LEGAL_DATABASE_URL")
        if db_url:
            if "?schema=" in db_url:
                db_url = db_url.split("?schema=")[0]
            _context.db_pool = pg_pool.SimpleConnectionPool(minconn=1, maxconn=10, dsn=db_url)
            logging.info("DB connection pool created")
        else:
            logging.warning("LEGAL_DATABASE_URL not set")
    except Exception as e:
        logging.warning(f"DB pool creation failed: {e}")
    Thread(target=cleanup_sessions, daemon=True).start()
    _load_user_functions(overwrite_globals=True)
    logging.info("Startup complete")

# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def get_session_state(session_id):
    if not session_id:
        raise HTTPException(status_code=400, detail="Session ID is required.")
    if session_id not in _context.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")
    return _context.sessions[session_id]

def init_openai_client(state, api_key, base_url=None):
    params = {}
    if base_url:
        params["base_url"] = base_url
    state.openai_client = OpenAI(api_key=api_key, **params)

# ---------------------------------------------------------------------------
# Persona detection
# ---------------------------------------------------------------------------

def process_persona(user_input: str):
    """Detect [legal ai] persona tag. Returns (persona, cleaned_input, filtered_tools, addendum_override)."""
    persona = "auto"
    filtered_tools = None
    addendum_override = None

    if user_input.lower().startswith("[legal ai]"):
        persona = "legal"
        user_input = user_input[10:].strip()
        legal_whitelist = ["search_legal", "summarize_legal_case", "get_legal_recommendation", "generate_legal_document"]
        filtered_tools = [t for t in _context.fun_manifest if t["function"]["name"] in legal_whitelist]
        try:
            with open("resources/prompts/legal_prompt.txt", "r", encoding="utf-8") as f:
                addendum_override = f.read()
        except Exception as e:
            logging.error(f"Failed to load legal_prompt.txt: {e}")

    return persona, user_input, filtered_tools, addendum_override

# ---------------------------------------------------------------------------
# Legal RAG persona flow
# ---------------------------------------------------------------------------

def run_legal_persona_ask(query: str) -> dict:
    request = RouterLegalAskRequest(query=query)
    result = legal_rag_ask(request)
    citations = result.get("citations", []) if isinstance(result, dict) else []
    source_metadata = [
        {
            "type": "legal_document",
            "title": c.get("title"),
            "category": c.get("category"),
            "bucket_slug": c.get("bucket_slug"),
            "year": c.get("year"),
            "source_url": c.get("source_url"),
            "s3_json_path": c.get("s3_json_path"),
            "snippet": c.get("snippet"),
            "full_text": c.get("full_text"),
            "relevance": 1.0,
        }
        for c in citations
    ]
    return {
        "answer": (result.get("answer", "") if isinstance(result, dict) else "").strip(),
        "source_metadata": source_metadata,
    }

# ---------------------------------------------------------------------------
# Citation helpers
# ---------------------------------------------------------------------------

def format_legal_citation_links(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    url_pattern = r"(https?://[^)]+|[^)]+)"
    def repl_law(m):
        return f'<a href="{m.group(2)}" class="legal-ref law">{m.group(1)}</a>'
    def repl_juris(m):
        return f'<a href="{m.group(2)}" class="legal-ref jurisprudence">{m.group(1)}</a>'
    text = re.sub(r"\[([^\]]+ Law)\]\(" + url_pattern + r"\)", repl_law, text)
    text = re.sub(r"\[([^\]]+ Jurisprudence)\]\(" + url_pattern + r"\)", repl_juris, text)
    return text

def repair_legal_source_links(text: str, search_results) -> str:
    if not text or not isinstance(text, str) or not search_results:
        return text
    source_ids = [str(r.get("item_id") or "").strip() for r in search_results if r.get("item_id")]
    if not source_ids:
        return text
    broken = re.compile(r"/sources/(?:\{[^)}]*\}|<[^>]*>|ACTUAL_ITEM_ID_HERE)?(?=[)\"\\s])")
    idx = [0]
    def repl(m):
        replacement = source_ids[min(idx[0], len(source_ids) - 1)]
        idx[0] += 1
        return f"/sources/{replacement}"
    return broken.sub(repl, text)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

generic_system_metaprompt = """
*** Function Calling Rule (CRITICAL - HIGHEST PRIORITY)
    You MUST call function immediately if query contains action intent.
    NO intermediate translation steps allowed.

*** Fresh Tool Invocation Rule (CRITICAL)
    If a tool is required for the current instruction, it MUST be invoked again
    unless the exact same function call is present in Memory Fact for the current loop.

*** Function Argument Authenticity Rule (CRITICAL)
    All required arguments MUST be explicitly provided by the user or deterministically
    derivable from current input. Never fabricate or guess arguments.

*** Function Failure Termination Rule
    If the same function call fails more than once consecutively, stop all further
    execution and report the failure clearly.
"""

generic_user_metaprompt = """
*** Language Consistency Rule (CRITICAL)
    ALL templates, headers, and structured output MUST match user's language.
    ZERO TOLERANCE for language mixing in ANY part of response.

*** Numerical Expression Formatting Rule (CRITICAL)
    Write all expressions entirely in plain text.
    DO NOT use LaTeX or symbols like $, \\times, \\frac.
    Use simple notation like x = 1, x^2 - 3x + 2, sqrt(2).
"""

# ---------------------------------------------------------------------------
# Message preparation
# ---------------------------------------------------------------------------

def calculate_tokens(text: str, model: str = None) -> int:
    m = model or _context.model
    try:
        enc = tiktoken.encoding_for_model("gpt-4" if "gpt-4" in m else "gpt-3.5-turbo")
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")
    return len(enc.encode(text))

def normalize_text_blocks(text: str) -> str:
    text = re.sub(r"(?m)^\s*$", "", text)
    text = re.sub(r"\n{2,}", "\n\n", text)
    return text.strip()

def summarize_conversation(state, new_history: str):
    combined = getattr(state, "summary", "") + "\n\n" + new_history
    try:
        resp = state.openai_client.chat.completions.create(
            model=_context.model,
            temperature=0.3,
            messages=[
                {"role": "system", "content": "Summarize the following conversation concisely."},
                {"role": "user", "content": combined},
            ],
            max_tokens=1024,
        )
        state.summary = "Summary: " + resp.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Conversation summary failed: {e}")

def prepare_chat_messages(state, query: str, addendum_override: str = None):
    prevs = ""
    if state.prompt:
        conversations = []
        for i, q in enumerate(state.prompt):
            if i < len(state.generated):
                conversations.append(f"User: {q.strip()}\nAssistant: {state.generated[i].strip()}")
            else:
                conversations.append(f"User: {q.strip()}")
        prevs = normalize_text_blocks("\n\n".join(conversations))

    if calculate_tokens(prevs) > 3000:
        recent = "\n\n".join(
            f"User: {q}\nAssistant: {a}"
            for q, a in zip(state.prompt[-2:], state.generated[-2:])
        )
        summarize_conversation(state, recent)
        prevs = state.summary

    context = ("\n\n[Past Conversation]\n" + prevs) if prevs else ""
    language, _ = langid.classify(query)

    system_content = ""
    if addendum_override:
        system_content += "[System Instructions]\n" + addendum_override.strip() + "\n\n"
        system_content += (
            "*** Function Calling Rule ***\n"
            "If query requires a specific action or tool, call the appropriate function immediately.\n"
            "Do NOT describe what you will do - actually call the function and show results.\n"
        )
    else:
        functions_enabled = bool(_context.fun_names)
        if functions_enabled:
            system_content += generic_system_metaprompt

    from datetime import datetime
    system_content += f"\n\n[Context Info]\nCurrent Date: {datetime.now().strftime('%A, %B %d, %Y')}\n"

    user_content = f"{context}\n\n[Instructions]\nUnless explicitly instructed otherwise, always respond in the `{language}`-locale language.\n"
    user_content += generic_user_metaprompt
    user_content = re.sub(r"\n\s*\n", "\n\n", user_content)
    user_content += f"\n\nQuestion:\n    {query}\n\nInstruction:\n    Please provide the most verbose and detailed response possible.\n"

    return [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content},
    ]

# ---------------------------------------------------------------------------
# Function execution
# ---------------------------------------------------------------------------

def clean_function_definitions(manifest_list: list) -> list:
    cleaned = []
    for item in manifest_list:
        if "name" in item and "parameters" in item:
            cleaned.append({k: v for k, v in item.items() if k != "strict"})
        elif "function" in item:
            cleaned.append({k: v for k, v in item["function"].items() if k != "strict"})
    return cleaned

def execute_function_call(function_call: dict, session_id: str = None):
    func_name = function_call.get("name")
    try:
        func_args = json.loads(function_call.get("arguments", "{}"))
    except Exception:
        func_args = {}

    if func_name not in globals():
        logging.error(f"Function {func_name} not found in globals.")
        return None

    try:
        result = globals()[func_name](**func_args)
        if func_name == "search_legal" and session_id and isinstance(result, dict):
            state = _context.sessions.get(session_id)
            if state is not None:
                state.last_search_legal_results = result.get("results", [])
        logging.info(f"Function {func_name} executed successfully.")
        return result
    except Exception as e:
        logging.error(f"Function {func_name} failed: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Reason loop (non-streaming, collects full response)
# ---------------------------------------------------------------------------

def run_function_chain(state, messages: list, max_chains: int = 7, session_id: str = None, tools: list = None):
    available_manifest = tools if tools is not None else _context.fun_manifest
    funcall_chains = []
    function_outputs = []
    full_response = ""

    def perform_chat(msgs):
        args = {
            "model": _context.model,
            "messages": msgs,
            "n": 1,
            "stream": True,
            "temperature": _context.temperature,
        }
        if available_manifest:
            args["functions"] = clean_function_definitions(available_manifest)
            args["function_call"] = "auto"
        return state.openai_client.chat.completions.create(**args)

    for _ in range(max_chains):
        function_call = {"name": None, "arguments": ""}
        stream_resp = perform_chat(messages)
        last_response = ""

        for chunk in stream_resp:
            delta = chunk.choices[0].delta
            if hasattr(delta, "function_call") and delta.function_call:
                fc = delta.function_call
                if fc.name:
                    function_call["name"] = fc.name
                if fc.arguments:
                    function_call["arguments"] += fc.arguments
            elif hasattr(delta, "content") and delta.content:
                last_response += delta.content.replace("~", "-")

        if last_response.strip():
            full_response = last_response.strip()

        if not function_call["name"]:
            break

        # HITL gate
        if not _context.auto_approval:
            return {"__hitl__": True, "function_call": function_call, "messages": messages, "tools": tools}

        # Duplicate check
        try:
            cur_args = json.loads(function_call["arguments"])
        except Exception:
            cur_args = function_call["arguments"]

        is_dup = any(
            fc["name"] == function_call["name"] and fc["args"] == cur_args
            for fc in funcall_chains
        )
        if is_dup:
            messages.append({"role": "system", "content": f"Function `{function_call['name']}` already called with same args. Do not repeat."})
            continue

        funcall_chains.append({"name": function_call["name"], "args": cur_args})
        result = execute_function_call(function_call, session_id=session_id)
        if result is None:
            continue
        function_outputs.append((function_call["name"], result))

        try:
            content = json.dumps(result, ensure_ascii=False)
        except Exception:
            content = str(result)
        messages.append({
            "role": "system",
            "content": (
                f"[Memory Fact]\nFunction `{function_call['name']}` returned:\n{content}\n\n"
                "Use this fact in all future reasoning. Do NOT re-call the same function."
            ),
        })
        messages.append({
            "role": "system",
            "content": (
                "[Constraints]\nIf a complete response has been produced, TERMINATE. "
                "Only execute new actions if their conditions are fully satisfied."
            ),
        })

    return full_response

def reason_loop(state, query: str, session_id: str = None, tools: list = None, addendum_override: str = None):
    messages = prepare_chat_messages(state, query, addendum_override=addendum_override)
    return run_function_chain(state, messages, session_id=session_id, tools=tools)

# ---------------------------------------------------------------------------
# Streaming reason loop (generator)
# ---------------------------------------------------------------------------

def streaming_run_function_chain(state, messages: list, max_chains: int = 7, session_id: str = None, tools: list = None):
    available_manifest = tools if tools is not None else _context.fun_manifest
    funcall_chains = []
    function_outputs = []
    full_response = ""

    def perform_chat(msgs):
        args = {
            "model": _context.model,
            "messages": msgs,
            "n": 1,
            "stream": True,
            "temperature": _context.temperature,
        }
        if available_manifest:
            args["functions"] = clean_function_definitions(available_manifest)
            args["function_call"] = "auto"
        return state.openai_client.chat.completions.create(**args)

    for _ in range(max_chains):
        function_call = {"name": None, "arguments": ""}
        stream_resp = perform_chat(messages)
        last_response = ""

        for chunk in stream_resp:
            delta = chunk.choices[0].delta
            if hasattr(delta, "function_call") and delta.function_call:
                fc = delta.function_call
                if fc.name:
                    function_call["name"] = fc.name
                if fc.arguments:
                    function_call["arguments"] += fc.arguments
            elif hasattr(delta, "content") and delta.content:
                part = delta.content.replace("~", "-")
                last_response += part
                yield part

        if last_response.strip():
            full_response = last_response.strip()

        if not function_call["name"]:
            break

        # HITL gate: emit pending_approval event and stop streaming
        if not _context.auto_approval:
            yield f"\n__HITL__{json.dumps({'function_call': function_call, 'messages': messages, 'tools': [t['function']['name'] for t in (tools or [])]})}"
            return

        try:
            cur_args = json.loads(function_call["arguments"])
        except Exception:
            cur_args = function_call["arguments"]

        is_dup = any(fc["name"] == function_call["name"] and fc["args"] == cur_args for fc in funcall_chains)
        if is_dup:
            messages.append({"role": "system", "content": f"Function `{function_call['name']}` already called. Do not repeat."})
            continue

        funcall_chains.append({"name": function_call["name"], "args": cur_args})
        yield f"[Tool] Executing `{function_call['name']}`...\n"

        result = execute_function_call(function_call, session_id=session_id)
        if result is None:
            continue
        function_outputs.append((function_call["name"], result))

        try:
            content = json.dumps(result, ensure_ascii=False)
        except Exception:
            content = str(result)
        messages.append({
            "role": "system",
            "content": f"[Memory Fact]\nFunction `{function_call['name']}` returned:\n{content}\n\nUse this fact. Do NOT re-call the same function.",
        })
        messages.append({
            "role": "system",
            "content": "[Constraints]\nIf a complete response has been produced, TERMINATE.",
        })

def streaming_reason_loop(state, query: str, session_id: str = None, tools: list = None, addendum_override: str = None):
    messages = prepare_chat_messages(state, query, addendum_override=addendum_override)
    yield from streaming_run_function_chain(state, messages, session_id=session_id, tools=tools)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/session-id")
async def get_new_session_id():
    new_id = str(uuid.uuid4())
    _context.sessions[new_id] = ChatState()
    return {"session_id": new_id}


@app.post("/chat")
def chat(request: ChatRequest):
    _t_start = time.time()
    session_id = request.session_id
    user_input = request.user_input or request.user_history_select or ""

    persona, user_input, filtered_tools, addendum_override = process_persona(user_input)

    if getattr(request, "document_context", None):
        doc_injection = f"\n\n[CONTEXT: The user is viewing the following document:]\n{request.document_context}"
        addendum_override = (addendum_override or "You are a helpful assistant.") + doc_injection

    if not user_input.strip():
        raise HTTPException(status_code=400, detail="User input is empty.")
    if not session_id or session_id not in _context.sessions:
        raise HTTPException(status_code=401, detail="Unknown session.")

    state = _context.sessions[session_id]
    state.last_used = time.time()

    if not _context.openai_api_key:
        raise HTTPException(status_code=400, detail="API key is required.")
    init_openai_client(state, _context.openai_api_key)

    # Direct legal RAG path
    if persona == "legal":
        try:
            legal_result = run_legal_persona_ask(user_input)
            state.prompt.append(user_input.strip())
            state.generated.append(legal_result["answer"])
            state.source_metadata = legal_result["source_metadata"]
            _context.sessions[session_id] = state
            logging.info("/chat [legal] %.2fs session=%s", time.time() - _t_start, session_id)
            return {
                "response": legal_result["answer"],
                "lookup": state.lookup,
                "source_metadata": state.source_metadata,
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Legal RAG ask failed: {str(e)}")

    # Normal path with optional HITL
    result = reason_loop(state, user_input, session_id=session_id, tools=filtered_tools, addendum_override=addendum_override)

    # HITL pending approval
    if isinstance(result, dict) and result.get("__hitl__"):
        fc = result["function_call"]
        state.pending_function_call = fc
        state.pending_messages = result["messages"]
        state.pending_tools = result["tools"]
        state.pending_addendum = addendum_override
        _context.sessions[session_id] = state
        try:
            args_parsed = json.loads(fc.get("arguments", "{}"))
        except Exception:
            args_parsed = {}
        logging.info("/chat [hitl:%s] %.2fs session=%s", fc["name"], time.time() - _t_start, session_id)
        return {
            "status": "pending_approval",
            "tool_name": fc["name"],
            "arguments": args_parsed,
            "intermediate_response": "",
            "hitl_decision": None,
        }

    state.prompt.append(user_input.strip())
    final_text = (result or "").strip()
    final_text = repair_legal_source_links(final_text, state.last_search_legal_results)
    if addendum_override and "LEGAL ASSISTANT MODE" in addendum_override:
        final_text = format_legal_citation_links(final_text)
    state.generated.append(final_text)
    _context.sessions[session_id] = state

    logging.info("/chat [%s] %.2fs session=%s", persona, time.time() - _t_start, session_id)
    return {
        "response": final_text,
        "lookup": state.lookup,
        "source_metadata": state.source_metadata,
    }


@app.post("/approve")
def approve(request: ApproveRequest):
    session_id = request.session_id
    decision = _normalize_hitl_decision(request.decision)

    if not session_id or session_id not in _context.sessions:
        raise HTTPException(status_code=404, detail="Session not found.")

    state = _context.sessions[session_id]

    if not state.pending_function_call:
        raise HTTPException(status_code=400, detail="No pending function call for this session.")

    fc = state.pending_function_call
    messages = state.pending_messages
    tools = state.pending_tools
    addendum_override = state.pending_addendum

    # Clear pending
    state.pending_function_call = None
    state.pending_messages = None
    state.pending_tools = None
    state.pending_addendum = None

    hitl_decision_record = {
        "tool_name": fc["name"],
        "decision": decision,
        "comments": request.comments,
    }

    if decision == "rejected":
        _context.sessions[session_id] = state
        return {
            "status": "rejected",
            "response": f"Action '{fc['name']}' was rejected.",
            "hitl_decision": hitl_decision_record,
        }

    if decision == "skipped_continue":
        # Skip the tool and let LLM continue without it
        messages.append({
            "role": "system",
            "content": f"The user skipped the call to `{fc['name']}`. Continue without it.",
        })
    else:
        # Execute the approved function
        result = execute_function_call(fc, session_id=session_id)
        try:
            content = json.dumps(result, ensure_ascii=False)
        except Exception:
            content = str(result)
        messages.append({
            "role": "system",
            "content": (
                f"[Memory Fact]\nFunction `{fc['name']}` (approved by user) returned:\n{content}\n\n"
                "Use this fact in all future reasoning."
            ),
        })
        messages.append({
            "role": "system",
            "content": "[Constraints]\nIf a complete response has been produced, TERMINATE.",
        })

    # Resume the reason loop with auto_approval temporarily True for this continuation
    _context.auto_approval = True
    try:
        available_manifest = [t for t in _context.fun_manifest if t["function"]["name"] in (tools or [])] if tools else _context.fun_manifest
        cont_result = run_function_chain(state, messages, session_id=session_id, tools=available_manifest)
    finally:
        _context.auto_approval = False

    if isinstance(cont_result, dict) and cont_result.get("__hitl__"):
        # Another tool call needs approval
        new_fc = cont_result["function_call"]
        state.pending_function_call = new_fc
        state.pending_messages = cont_result["messages"]
        state.pending_tools = cont_result["tools"]
        state.pending_addendum = addendum_override
        _context.sessions[session_id] = state
        try:
            args_parsed = json.loads(new_fc.get("arguments", "{}"))
        except Exception:
            args_parsed = {}
        return {
            "status": "pending_approval",
            "tool_name": new_fc["name"],
            "arguments": args_parsed,
            "intermediate_response": "",
            "hitl_decision": hitl_decision_record,
        }

    final_text = (cont_result or "").strip()
    final_text = repair_legal_source_links(final_text, state.last_search_legal_results)
    if addendum_override and "LEGAL ASSISTANT MODE" in addendum_override:
        final_text = format_legal_citation_links(final_text)
    state.generated.append(final_text)
    _context.sessions[session_id] = state

    return {
        "status": "completed",
        "response": final_text,
        "hitl_decision": hitl_decision_record,
    }


def _normalize_hitl_decision(decision: str) -> str:
    aliases = {
        "a": "approved", "approve": "approved", "approved": "approved", "yes": "approved", "y": "approved",
        "r": "rejected", "reject": "rejected", "rejected": "rejected", "no": "rejected", "n": "rejected",
        "s": "skipped_continue", "skip": "skipped_continue", "skipped": "skipped_continue",
        "continue": "skipped_continue", "skipped_continue": "skipped_continue", "skip_continue": "skipped_continue",
        "skipped_by_user_and_continued": "skipped_continue",
    }
    normalized = str(decision).strip().lower().replace("-", "_").replace("/", "_")
    if normalized in aliases:
        return aliases[normalized]
    return "approved"


@app.post("/set-hitl")
def set_hitl(request: SetHitlRequest):
    _context.auto_approval = request.auto_approval
    return {"message": f"HITL auto_approval set to {request.auto_approval}", "auto_approval": _context.auto_approval}


@app.get("/hitl-status")
def hitl_status():
    return {"auto_approval": _context.auto_approval}


@app.websocket("/chat-stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            raw_data = await websocket.receive_text()
            try:
                data = json.loads(raw_data)
            except Exception as e:
                await websocket.send_text(f"[Error] Invalid JSON: {e}")
                continue

            msg_type = data.get("type", "chat")

            if msg_type == "approve":
                session_id = data.get("session_id")
                decision = data.get("decision", "approved")
                comments = data.get("comments")
                if not session_id or session_id not in _context.sessions:
                    await websocket.send_text("[Error] Unknown session.")
                    await websocket.send_text(_context.__END__)
                    continue

                state = _context.sessions[session_id]
                if not state.pending_function_call:
                    await websocket.send_text("[Error] No pending function call.")
                    await websocket.send_text(_context.__END__)
                    continue

                decision = _normalize_hitl_decision(decision)
                fc = state.pending_function_call
                messages = state.pending_messages
                tools = state.pending_tools
                addendum_override = state.pending_addendum

                state.pending_function_call = None
                state.pending_messages = None
                state.pending_tools = None
                state.pending_addendum = None

                if decision == "rejected":
                    await websocket.send_text(f"Action '{fc['name']}' was rejected.")
                    await websocket.send_text(_context.__END__)
                    continue

                if decision == "skipped_continue":
                    messages.append({"role": "system", "content": f"User skipped `{fc['name']}`. Continue without it."})
                else:
                    result = execute_function_call(fc, session_id=session_id)
                    try:
                        content = json.dumps(result, ensure_ascii=False)
                    except Exception:
                        content = str(result)
                    messages.append({
                        "role": "system",
                        "content": f"[Memory Fact]\nFunction `{fc['name']}` returned:\n{content}\n\nUse this fact.",
                    })

                available_manifest = _context.fun_manifest
                _context.auto_approval = True
                full_response = ""
                try:
                    for chunk in streaming_run_function_chain(state, messages, session_id=session_id, tools=available_manifest):
                        if chunk.startswith("__HITL__"):
                            hitl_data = json.loads(chunk[8:])
                            new_fc = hitl_data["function_call"]
                            state.pending_function_call = new_fc
                            state.pending_messages = hitl_data["messages"]
                            state.pending_tools = hitl_data.get("tools")
                            state.pending_addendum = addendum_override
                            _context.sessions[session_id] = state
                            try:
                                args_parsed = json.loads(new_fc.get("arguments", "{}"))
                            except Exception:
                                args_parsed = {}
                            await websocket.send_text(json.dumps({
                                "status": "pending_approval",
                                "tool_name": new_fc["name"],
                                "arguments": args_parsed,
                            }))
                            break
                        await websocket.send_text(chunk)
                        full_response += chunk
                finally:
                    _context.auto_approval = False

                if full_response:
                    final_text = full_response.strip()
                    final_text = repair_legal_source_links(final_text, state.last_search_legal_results)
                    if addendum_override and "LEGAL ASSISTANT MODE" in addendum_override:
                        final_text = format_legal_citation_links(final_text)
                    state.generated.append(final_text)
                _context.sessions[session_id] = state
                await websocket.send_text(_context.__END__)
                continue

            # Regular chat message
            request = ChatRequest(**{k: v for k, v in data.items() if k in ChatRequest.model_fields})
            session_id = request.session_id

            if not session_id or session_id not in _context.sessions:
                await websocket.send_text("[Error] Unknown session.")
                await websocket.send_text(_context.__END__)
                continue

            state = _context.sessions[session_id]
            state.last_used = time.time()
            init_openai_client(state, _context.openai_api_key)

            user_input = request.user_input or getattr(request, "user_history_select", "") or ""
            persona, user_input, filtered_tools, addendum_override = process_persona(user_input)

            if getattr(request, "document_context", None):
                doc_injection = f"\n\n[CONTEXT: User is viewing:]\n{request.document_context}"
                addendum_override = (addendum_override or "You are a helpful assistant.") + doc_injection

            if not user_input.strip():
                await websocket.send_text("[Error] User input is empty.")
                await websocket.send_text(_context.__END__)
                continue

            full_response = ""
            _ws_t_start = time.time()
            try:
                if persona == "legal":
                    legal_result = await asyncio.get_event_loop().run_in_executor(
                        None, run_legal_persona_ask, user_input
                    )
                    state.source_metadata = legal_result["source_metadata"]
                    if state.source_metadata:
                        await websocket.send_text(f"[Sources] {json.dumps(state.source_metadata)}")
                    await websocket.send_text(legal_result["answer"])
                    await websocket.send_text(_context.__END__)
                    state.prompt.append(user_input)
                    state.generated.append(legal_result["answer"])
                    _context.sessions[session_id] = state
                    logging.info("/chat-stream [legal] %.2fs session=%s", time.time() - _ws_t_start, session_id)
                    continue

                for chunk in streaming_reason_loop(state, user_input, session_id=session_id, tools=filtered_tools, addendum_override=addendum_override):
                    if chunk.startswith("__HITL__"):
                        hitl_data = json.loads(chunk[8:])
                        fc = hitl_data["function_call"]
                        state.pending_function_call = fc
                        state.pending_messages = hitl_data["messages"]
                        state.pending_tools = hitl_data.get("tools")
                        state.pending_addendum = addendum_override
                        _context.sessions[session_id] = state
                        try:
                            args_parsed = json.loads(fc.get("arguments", "{}"))
                        except Exception:
                            args_parsed = {}
                        await websocket.send_text(json.dumps({
                            "status": "pending_approval",
                            "tool_name": fc["name"],
                            "arguments": args_parsed,
                        }))
                        break
                    await websocket.send_text(chunk)
                    full_response += chunk

                if full_response:
                    state.prompt.append(user_input)
                    final_text = full_response.strip()
                    final_text = repair_legal_source_links(final_text, state.last_search_legal_results)
                    if addendum_override and "LEGAL ASSISTANT MODE" in addendum_override:
                        final_text = format_legal_citation_links(final_text)
                    state.generated.append(final_text)
                    _context.sessions[session_id] = state
                    await websocket.send_text(_context.__END__)

            except Exception as e:
                await websocket.send_text(f"[Error] {e}")
                await websocket.send_text(_context.__END__)

    except WebSocketDisconnect:
        logging.debug("WebSocket connection closed.")


@app.post("/install-embeddings")
async def install_embeddings(file: UploadFile = File(...), session_id: str = None):
    try:
        content = await file.read()
        with open("embeddings.pkz", "wb") as f:
            f.write(content)
        return {"message": "Embeddings file saved successfully."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save embeddings: {e}")


@app.post("/install-user-functions")
async def install_user_functions(zip_file: UploadFile = File(...)):
    try:
        content = await zip_file.read()
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(content)) as zf:
                zf.extractall(tmp)
                names = zf.namelist()
            py_files = [f for f in names if f.endswith(".py")]
            manifest_files = [f for f in names if f.endswith(".manifest")]
            env_files = [f for f in names if f.endswith(".env")]
            req_files = [f for f in names if f.endswith("requirements.txt")]
            if not py_files or not manifest_files:
                raise HTTPException(status_code=400, detail="ZIP must include .py and .manifest files.")
            shutil.copy(os.path.join(tmp, py_files[0]), os.path.join(_context.FUNCTIONS_DIR, "user_functions.py"))
            shutil.copy(os.path.join(tmp, manifest_files[0]), os.path.join(_context.FUNCTIONS_DIR, "user_functions.manifest"))
            if env_files:
                shutil.copy(os.path.join(tmp, env_files[0]), os.path.join(_context.FUNCTIONS_DIR, "user_functions.env"))
            if req_files:
                req_dst = os.path.join(_context.FUNCTIONS_DIR, "requirements.txt")
                shutil.copy(os.path.join(tmp, req_files[0]), req_dst)
                subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_dst], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        _load_user_functions(overwrite_globals=False)
        return {"message": "User functions installed successfully."}
    except zipfile.BadZipFile:
        raise HTTPException(status_code=400, detail="Invalid ZIP file.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error installing user functions: {e}")


@app.post("/export")
def export_chat(request: ExportRequest):
    state = get_session_state(request.session_id)
    state.last_used = time.time()
    df = pd.DataFrame({"Prompt": state.prompt, "Response": state.generated})
    file_type = request.file_type.lower()
    if file_type == "csv":
        buf = io.BytesIO()
        buf.write(df.to_csv(index=False).encode("utf-8"))
        buf.seek(0)
        mime = "text/csv"
    else:
        html = '<meta charset="utf-8">\n' + df.to_html(index=False, escape=False)
        buf = io.BytesIO(html.encode("utf-8"))
        mime = "text/html"
    return StreamingResponse(buf, media_type=mime, headers={"Content-Disposition": f"attachment;filename={request.file_name}.{file_type}"})


@app.post("/import")
def import_chat(request: ImportRequest):
    state = get_session_state(request.session_id)
    state.last_used = time.time()
    try:
        df = pd.read_csv(io.StringIO(request.conversation), encoding="utf-8")
    except Exception:
        df = pd.read_csv(io.StringIO(request.conversation), encoding="cp949")
    state.prompt = df["Prompt"].tolist()
    state.generated = df["Response"].tolist()
    _context.sessions[request.session_id] = state
    return {"message": "Chat imported successfully."}

# ---------------------------------------------------------------------------
# User functions loader
# ---------------------------------------------------------------------------

def safe_load_json(path: str) -> list:
    for enc in _context.char_encodings:
        try:
            with open(path, encoding=enc) as f:
                return json.load(f)
        except Exception:
            continue
    raise ValueError(f"Unable to read JSON: {path}")

def _load_user_functions(overwrite_globals: bool = False):
    if not hasattr(_context, "user_functions"):
        _context.user_functions = {}
    if not hasattr(_context, "fun_manifest"):
        _context.fun_manifest = []
    if not hasattr(_context, "fun_names"):
        _context.fun_names = []

    # Built-in execute_code tool
    def execute_code(code: str):
        import ast, threading as _threading
        holder = {}
        def _run(c):
            try:
                import builtins as _b
                allowed = {k: getattr(_b, k) for k in ["print","str","int","float","bool","list","dict","set","tuple","abs","min","max","sum","round","range","len","sorted","__import__"]}
                g = {"__builtins__": allowed}
                l = {}
                parsed = ast.parse(c, mode="exec")
                if parsed.body and isinstance(parsed.body[-1], ast.Expr):
                    c = c.rstrip() + f"\n_result_ = {ast.unparse(parsed.body[-1])}"
                exec(c, g, l)
                holder["result"] = l.get("_result_", "Code executed successfully (no explicit result).")
                holder["status"] = "success"
            except Exception:
                import traceback
                holder["result"] = traceback.format_exc()
                holder["status"] = "error"
        t = _threading.Thread(target=_run, args=(code,))
        t.start()
        t.join(timeout=10)
        if t.is_alive():
            return {"status": "error", "result": "Code execution timed out."}
        return holder

    ec_manifest = {
        "type": "function",
        "function": {
            "name": "execute_code",
            "description": "Execute a short Python expression and return its result.",
            "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python code to execute."}}, "required": ["code"]},
        },
    }
    if "execute_code" not in _context.fun_names:
        _context.fun_manifest.append(ec_manifest)
        _context.fun_names.append("execute_code")
        _context.user_functions["execute_code"] = execute_code
        globals()["execute_code"] = execute_code

    functions_dir = os.path.join("resources", "functions")
    py_path = os.path.join(functions_dir, "user_functions.py")
    manifest_path = os.path.join(functions_dir, "user_functions.manifest")
    env_path = os.path.join(functions_dir, "user_functions.env")
    req_path = os.path.join(functions_dir, "requirements.txt")

    if os.path.exists(env_path):
        try:
            dotenv.load_dotenv(dotenv_path=env_path, override=True)
        except Exception as e:
            logging.warning(f"user_functions.env load failed: {e}")

    if os.path.exists(req_path):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", req_path], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except Exception as e:
            logging.warning(f"requirements.txt install failed: {e}")

    if os.path.exists(py_path):
        try:
            spec = importlib.util.spec_from_file_location("resources.functions.user_functions", py_path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            funcs = {n: f for n, f in vars(mod).items() if callable(f) and not n.startswith("__")}
            if overwrite_globals:
                for n, f in funcs.items():
                    globals()[n] = f
            else:
                _context.user_functions.update(funcs)
                for n, f in funcs.items():
                    globals()[n] = f
            logging.info(f"User functions loaded: {list(funcs.keys())}")
        except Exception as e:
            logging.warning(f"User functions load failed: {e}")

    if os.path.exists(manifest_path):
        try:
            loaded = safe_load_json(manifest_path)
            # Reset and rebuild (keep execute_code)
            _context.fun_manifest = [ec_manifest] + [item for item in loaded if item.get("function", {}).get("name") != "execute_code"]
            _context.fun_names = [item["function"]["name"] for item in _context.fun_manifest]
            logging.info(f"Manifest loaded: {_context.fun_names}")
        except Exception as e:
            logging.warning(f"Manifest load failed: {e}")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_context.model = os.getenv("CHAT_MODEL", "gpt-4o-mini")
_context.temperature = 1.0
_context.informed = True
_context.show_clues = []
_context.expertise = "General"
_context.tone = "factual"

# ---------------------------------------------------------------------------
# Legal Case Search & Detail Endpoints
# ---------------------------------------------------------------------------

class LegalSearchRequest(BaseModel):
    prompt: str = ""
    page: int = 1
    limit: int = 5
    optimized_query: Optional[str] = None  # pass on page 2+ to skip re-optimizing
    content_types: List[str] = None

_LEGAL_SEARCH_MAX_POOL = 50  # max results fetched from RAG per query

@app.post("/api/legal/search")
async def api_legal_search(request: LegalSearchRequest):
    try:
        prompt = request.prompt.strip()
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required.")
        if not _context.openai_api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key is not configured.")

        page = max(1, request.page)
        limit = max(1, min(request.limit, 20))
        offset = (page - 1) * limit

        # Reuse optimized_query on page 2+ to avoid an extra GPT call
        if request.optimized_query:
            optimized_query = request.optimized_query.strip()
        else:
            client = OpenAI(api_key=_context.openai_api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert Philippine legal researcher. "
                            "Extract the core legal issue, relevant keywords, or specific laws from the user prompt "
                            "to create a concise search query (max 10 words) optimized for semantic vector search. "
                            "Return ONLY the search string, nothing else."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            optimized_query = response.choices[0].message.content.strip()

        logging.info(f"[Legal Search] page={page} '{prompt}' -> '{optimized_query}'")

        rag_result = legal_rag_search(query=optimized_query, limit=_LEGAL_SEARCH_MAX_POOL)
        rag_rows = rag_result.get("results", []) if isinstance(rag_result, dict) else []

        all_results = [
            {
                **row,
                "item_id": str(row.get("id")) if row.get("id") is not None else None,
                "text_content": row.get("snippet", ""),
                "metadata": {
                    "category": row.get("category"),
                    "bucket_slug": row.get("bucket_slug"),
                    "year": row.get("year"),
                    "source_url": row.get("source_url"),
                    "s3_json_path": row.get("s3_json_path"),
                },
            }
            for row in rag_rows
        ]

        total = len(all_results)
        paged = all_results[offset: offset + limit]
        total_pages = (total + limit - 1) // limit if total > 0 else 1

        return {
            "success": True,
            "query": prompt,
            "ai_optimized_query": optimized_query,
            "page": page,
            "limit": limit,
            "total_results": total,
            "total_pages": total_pages,
            "has_next": page < total_pages,
            "has_prev": page > 1,
            "results": paged,
            "search_type": "hybrid_rag",
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[Legal Search] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/legal/case/{item_id}")
async def api_legal_case_detail(item_id: str):
    try:
        if not str(item_id).isdigit():
            raise HTTPException(status_code=400, detail="item_id must be a numeric legal document id.")

        doc = legal_rag_get_document(int(item_id))
        if not isinstance(doc, dict):
            raise HTTPException(status_code=404, detail=f"Case '{item_id}' not found.")

        metadata = doc.get("metadata_json") or {}
        return {
            "id": doc.get("id"),
            "item_id": str(doc.get("id")),
            "type": doc.get("category"),
            "title": doc.get("title"),
            "url": doc.get("source_url"),
            "text_content": doc.get("full_text") or doc.get("summary") or doc.get("concise_summary") or "",
            "gr_number": metadata.get("gr_number", ""),
            "law_number": metadata.get("law_number", ""),
            "date": metadata.get("date", ""),
            "year": doc.get("year", ""),
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[Legal Case Detail] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Document Analyzer Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/legal/document-upload-url")
async def generate_document_upload_url(request: DocumentUploadUrlRequest):
    """Generate an S3 presigned PUT URL so the frontend can upload directly to S3."""
    safe_filename = "".join(c for c in request.filename if c.isalnum() or c in " ._-")
    unique_id = str(uuid.uuid4())
    s3_key = f"uploads/documents/{unique_id}-{safe_filename}"
    content_type = request.content_type or "application/octet-stream"

    presigned_url = s3_storage.generate_presigned_put(s3_key, content_type=content_type)
    if not presigned_url:
        raise HTTPException(status_code=500, detail="Failed to generate S3 upload URL. Is S3 configured?")

    return {"success": True, "s3_key": s3_key, "url": presigned_url, "content_type": content_type}


@app.post("/api/legal/analyze-document")
async def analyze_legal_document(request: AnalyzeS3DocumentRequest):
    """Download a document from S3, extract its text, and return a structured AI legal analysis."""
    s3_key = request.s3_key
    filename = request.filename or os.path.basename(s3_key)
    ext = os.path.splitext(filename)[1].lower()

    tmp_dir = tempfile.gettempdir()
    local_path = os.path.join(tmp_dir, os.path.basename(s3_key))

    downloaded = s3_storage.download_from_s3(s3_key, local_path)
    if not downloaded:
        raise HTTPException(status_code=404, detail="File not found in S3 or download failed.")

    extracted_text = ""

    try:
        with open(local_path, "rb") as f:
            contents = f.read()

        MAX_SIZE = 20 * 1024 * 1024
        if len(contents) > MAX_SIZE:
            raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 20MB.")

        if ext == ".txt":
            for enc in ["utf-8", "cp1252", "latin-1"]:
                try:
                    extracted_text = contents.decode(enc)
                    break
                except UnicodeDecodeError:
                    continue
            if not extracted_text:
                raise HTTPException(status_code=400, detail="Could not decode text file.")

        elif ext == ".pdf":
            try:
                import PyPDF2
                pdf_reader = PyPDF2.PdfReader(io.BytesIO(contents))
                pages = [page.extract_text() for page in pdf_reader.pages if page.extract_text()]
                extracted_text = "\n\n".join(p.strip() for p in pages)
            except ImportError:
                try:
                    import pdfplumber
                    with pdfplumber.open(io.BytesIO(contents)) as pdf:
                        extracted_text = "\n\n".join(p.extract_text() or "" for p in pdf.pages).strip()
                except ImportError:
                    raise HTTPException(status_code=500, detail="PDF library not installed. Run: pip install PyPDF2")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {e}")

        elif ext in (".docx", ".doc"):
            try:
                import docx
                doc = docx.Document(io.BytesIO(contents))
                extracted_text = "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
            except ImportError:
                raise HTTPException(status_code=500, detail="DOCX library not installed. Run: pip install python-docx")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to parse DOCX: {e}")

        elif ext in (".mp3", ".wav", ".m4a"):
            if not _context.openai_api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key required for audio transcription.")
            try:
                client = OpenAI(api_key=_context.openai_api_key)
                with open(local_path, "rb") as audio_file:
                    transcription = client.audio.transcriptions.create(model="whisper-1", file=audio_file)
                extracted_text = transcription.text
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to transcribe audio: {e}")

        elif ext in (".png", ".jpg", ".jpeg"):
            if not _context.openai_api_key:
                raise HTTPException(status_code=400, detail="OpenAI API key required for image OCR.")
            try:
                import base64, mimetypes
                b64 = base64.b64encode(contents).decode("utf-8")
                mime_type = mimetypes.guess_type(filename)[0] or f"image/{ext[1:]}"
                client = OpenAI(api_key=_context.openai_api_key)
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[{"role": "user", "content": [
                        {"type": "text", "text": "Extract all the text from this image exactly as written. If there is no text, describe the image briefly."},
                        {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{b64}"}},
                    ]}],
                    max_tokens=3000,
                )
                extracted_text = response.choices[0].message.content.strip()
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to process image OCR: {e}")

        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type '{ext}'. Supported: PDF, DOCX, TXT, PNG, JPG, MP3, WAV, M4A.")

        extracted_text = extracted_text.strip()
        if not extracted_text:
            raise HTTPException(status_code=400, detail="No text could be extracted from the document.")

        CHAR_LIMIT = 50000
        truncated = len(extracted_text) > CHAR_LIMIT
        if truncated:
            extracted_text = extracted_text[:CHAR_LIMIT]

        logging.info(f"[Analyze Document] Extracted {len(extracted_text)} chars from '{filename}'")

        ai_summary = None
        if _context.openai_api_key:
            try:
                client = OpenAI(api_key=_context.openai_api_key)
                system_prompt = (
                    "You are an expert Philippine legal document analyst with deep knowledge of Philippine law, "
                    "jurisprudence, and the Civil Code, Revised Penal Code, Labor Code, and Supreme Court decisions. "
                    "Analyze the provided legal document and produce a comprehensive legal analysis with the following sections. "
                    "Use bullet points and sub-points where appropriate.\n\n"
                    "## 1. Document Overview\n"
                    "Identify the document type, parties involved, date, jurisdiction, and overall legal purpose.\n\n"
                    "## 2. Key Legal Issues & Provisions\n"
                    "List all significant legal points, obligations, rights, conditions, and prohibitions with their implications.\n\n"
                    "## 3. Relevant Philippine Laws & Jurisprudence\n"
                    "Cite applicable statutes (Civil Code Articles, Labor Code provisions, RA numbers) and Supreme Court decisions (G.R. numbers).\n\n"
                    "## 4. Notable Clauses or Concerns\n"
                    "Highlight unusual, ambiguous, or potentially disadvantageous clauses and explain the legal risk.\n\n"
                    "## 5. Parties' Rights & Obligations\n"
                    "Summarize what each named party is entitled to and obligated to do.\n\n"
                    "## 6. Potential Legal Issues or Disputes\n"
                    "Identify scenarios that could lead to disputes or enforcement problems and how to mitigate them.\n\n"
                    "## 7. Recommendations\n"
                    "Provide specific, actionable legal advice: what to negotiate, watch out for, and suggested next steps.\n\n"
                    "Be thorough and detailed. This analysis will be used by a lawyer or client seeking legal guidance."
                )
                text_for_summary = extracted_text[:25000]
                if truncated:
                    text_for_summary += f"\n\n[Note: Document was truncated — only the first {CHAR_LIMIT:,} characters were analyzed.]"

                response = client.chat.completions.create(
                    model=_context.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Document: **{filename}**\n\n---\n\n{text_for_summary}"},
                    ],
                    temperature=0.2,
                    max_tokens=4000,
                )
                ai_summary = response.choices[0].message.content.strip()
                logging.info(f"[Analyze Document] AI summary generated for '{filename}' ({len(ai_summary)} chars)")
            except Exception as e:
                logging.warning(f"[Analyze Document] AI summary failed (non-fatal): {e}")

        file_url = s3_storage.generate_presigned_get(s3_key)

        return {
            "success": True,
            "filename": filename,
            "s3_key": s3_key,
            "file_url": file_url,
            "ai_summary": ai_summary,
            "char_count": len(extracted_text),
            "truncated": truncated,
        }

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[Analyze Document] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")
    finally:
        if os.path.exists(local_path):
            try:
                os.remove(local_path)
            except Exception as cleanup_err:
                logging.warning(f"Failed to clean up temp file {local_path}: {cleanup_err}")


@app.post("/api/legal/upload-and-analyze")
async def upload_and_analyze(file: UploadFile = File(...)):
    """Upload a document directly and get back an AI legal analysis in one step. Useful for testing."""
    contents = await file.read()

    if len(contents) > 20 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum allowed size is 20MB.")

    filename = file.filename or "document"
    safe_filename = "".join(c for c in filename if c.isalnum() or c in " ._-")
    s3_key = f"uploads/documents/{uuid.uuid4()}-{safe_filename}"

    uploaded = s3_storage.upload_bytes_to_s3(contents, s3_key, content_type=file.content_type or "application/octet-stream")
    if not uploaded:
        raise HTTPException(status_code=500, detail="Failed to upload file to S3. Is S3 configured?")

    # Reuse the analyze endpoint logic by calling it internally
    from pydantic import BaseModel as _BM
    class _Req(_BM):
        s3_key: str
        filename: Optional[str] = None
        session_id: Optional[str] = None

    return await analyze_legal_document(_Req(s3_key=s3_key, filename=filename))


@app.post("/api/legal/synthesize-documents")
async def synthesize_documents(request: SynthesizeDocumentsRequest):
    """Cross-document synthesis: takes multiple AI summaries and produces a unified strategic analysis."""
    if not request.summaries:
        raise HTTPException(status_code=400, detail="No summaries provided for synthesis.")
    if not _context.openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    client = OpenAI(api_key=_context.openai_api_key)

    system_prompt = """
You are an expert Philippine legal analyst and strategic advisor. The user has uploaded MULTIPLE legal documents.
The text below contains individual AI analyses for each document.

Provide a "Level 2" Cross-Document Synthesis with these sections:

## Cross-Document Overview
How these documents relate to each other overall.

## Common Themes & Connections
Consistent obligations, rights, or themes running across documents.

## Conflicts & Discrepancies
Contradictions or conflicts between documents. If none, state they appear aligned.

## Aggregated Risk Assessment
The biggest legal vulnerabilities or risks across the entire package of documents.

## Unified Strategic Recommendations
A single prioritized list of actionable next steps based on the combined context.

Be legally precise, referencing Philippine law where applicable. Synthesize — do not regurgitate summaries verbatim.
"""

    combined_text = ""
    for i, summary in enumerate(request.summaries):
        combined_text += f"=== DOCUMENT {i + 1} ANALYSIS ===\n{summary}\n\n"

    try:
        response = client.chat.completions.create(
            model=_context.model,
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {"role": "user", "content": combined_text.strip()},
            ],
            temperature=0.2,
            max_tokens=3000,
        )
        synthesis = response.choices[0].message.content.strip()
        logging.info(f"[Synthesize Documents] Synthesis generated for {len(request.summaries)} documents.")
        return {"success": True, "synthesis": synthesis}
    except Exception as e:
        logging.error(f"[Synthesize Documents] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate synthesis: {e}")
