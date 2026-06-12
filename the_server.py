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
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, Response
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
    get_cached_db as legal_rag_get_cached_db,
)
from legal_rag.markdown_format import format_document_combined, prepend_title_heading
import s3_storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ---------------------------------------------------------------------------
# API Context (global state)
# ---------------------------------------------------------------------------

class ApiContext:
    embeddings = None
    addendum = ""
    fun_manifest: list = []      # general tools only (no persona tag)
    all_fun_manifest: list = []  # all tools including persona-specific
    fun_names: list = []
    user_functions: dict = {}
    sessions: dict = {}
    db_pool = None
    auto_approval: bool = False  # HITL: when True all tool calls execute without asking
    xai_reason_visible: bool = False  # XAI: when True, REASON line is passed through to clients

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
_context.legal_library_url: str = os.getenv("LEGAL_LIBRARY_URL", "").rstrip("/")
_context.temperature: float = 1.0
_context.informed: bool = True
_context.show_clues: list = []
_context.expertise: str = "General"
_context.tone: str = "factual"
_context.build_marker: str = os.getenv("APP_BUILD_MARKER", "legal-citation-guard-v1")

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
        self.last_garment_result: dict = {}
        self.last_cosmetics_result: dict = {}
        self.last_maps_result: list = []
        self.last_nav_result: dict = {}
        self.last_tailor_result: dict = {}
        self.last_outfit_ids_result: list = []
        self.confirmed_gender: str = ""
        self.sitemap_context: list = []
        self.openai_client = None
        self.last_used: float = time.time()
        self.user_id: Optional[str] = None
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
# Glass-box trace broadcast (SCL — Supervised Cognitive Loop)
# ---------------------------------------------------------------------------

_trace_queues: set = set()
_app_event_loop = None  # captured at startup so worker threads can schedule on the main loop
_metric_counters: dict = {}

def broadcast_trace(event_type: str, text: str, session_id: str = None, summary: str = None):
    data = json.dumps({"type": event_type, "text": text.strip(), "summary": summary.strip() if summary else None, "session_id": session_id, "ts": time.time()})
    def _put_all():
        for q in list(_trace_queues):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass
    if _app_event_loop is not None and _app_event_loop.is_running():
        _app_event_loop.call_soon_threadsafe(_put_all)
    else:
        _put_all()


def increment_metric_counter(name: str, value: int = 1, tags: dict = None, session_id: str = None):
    """Emit a lightweight counter metric via logs + optional trace."""
    if not name:
        return
    key = (name, tuple(sorted((tags or {}).items())))
    _metric_counters[key] = _metric_counters.get(key, 0) + int(value)
    logging.info(
        "[metric] type=counter name=%s value=%s total=%s tags=%s",
        name,
        value,
        _metric_counters[key],
        tags or {},
    )
    # CloudWatch Embedded Metric Format (EMF) for native metrics ingestion.
    try:
        dimensions = ["metric_name"]
        emf_payload = {
            "_aws": {
                "Timestamp": int(time.time() * 1000),
                "CloudWatchMetrics": [
                    {
                        "Namespace": "ChatWonder/Legal",
                        "Dimensions": [dimensions],
                        "Metrics": [{"Name": name, "Unit": "Count"}],
                    }
                ],
            },
            "metric_name": name,
            name: int(value),
            "counter_total": _metric_counters[key],
        }
        if tags:
            for k, v in tags.items():
                safe_key = re.sub(r"[^A-Za-z0-9_]", "_", str(k))
                if safe_key and safe_key not in emf_payload:
                    emf_payload[safe_key] = str(v)
                    dimensions.append(safe_key)
        logging.info(json.dumps(emf_payload, ensure_ascii=True))
    except Exception as e:
        logging.warning("[metric] EMF emit failed name=%s err=%s", name, e)
    tracer = globals().get("broadcast_trace")
    if callable(tracer):
        try:
            tracer(
                "metric",
                f"counter {name} +{value} total={_metric_counters[key]} tags={tags or {}}",
                session_id,
            )
        except Exception:
            # Metrics must never break response generation.
            pass

@app.get("/trace-stream", summary="Glass-box trace SSE stream (consumed by scl-core-v2)")
async def trace_stream():
    async def _generator():
        q: asyncio.Queue = asyncio.Queue(maxsize=200)
        _trace_queues.add(q)
        try:
            yield "data: {\"type\":\"connected\"}\n\n"
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=25)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield "data: {\"type\":\"ping\"}\n\n"
        except asyncio.CancelledError:
            pass
        finally:
            _trace_queues.discard(q)
    return StreamingResponse(_generator(), media_type="text/event-stream",
                             headers={
                                 "Cache-Control": "no-cache",
                                 "X-Accel-Buffering": "no",
                                 "Access-Control-Allow-Origin": "*",
                             })

# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_input: str = ""
    user_history_select: Optional[str] = None
    session_id: Optional[str] = None
    document_context: Optional[str] = None
    weather: Optional[dict] = None
    skin_analysis: Optional[dict] = None
    location: Optional[dict] = None
    sitemap_context: Optional[list] = None
    user_id: Optional[str] = None
    gender: Optional[str] = None
    category: Optional[dict] = None
    sets: Optional[int] = None

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

class CosmeticScanRequest(BaseModel):
    front_s3_key: Optional[str] = None
    back_s3_key: str
    skin_type: Optional[str] = "general"
    session_id: Optional[str] = None

class CosmeticMatchRequest(BaseModel):
    product_a_s3_key: str
    product_b_s3_key: str
    skin_type: Optional[str] = "general"
    session_id: Optional[str] = None

# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event():
    global _app_event_loop
    _app_event_loop = asyncio.get_event_loop()
    logging.info("Starting Chat Wonder v2...")
    try:
        from psycopg2 import pool as pg_pool
        from urllib.parse import unquote
        db_url = os.getenv("LEGAL_DATABASE_URL")
        if db_url:
            if "?schema=" in db_url:
                db_url = db_url.split("?schema=")[0]
            if db_url.startswith(("postgres://", "postgresql://")):
                # URI format — urlparse mishandles passwords with encoded brackets/colons, so split manually.
                # Safe because @ in passwords must be encoded as %40.
                scheme_userinfo, hostinfo = db_url.rsplit("@", 1)
                userinfo = scheme_userinfo.split("://", 1)[1]
                db_user, db_pass_enc = userinfo.split(":", 1)
                host_port, dbname = hostinfo.split("/", 1)
                db_host, db_port = host_port.rsplit(":", 1)
                _context.db_pool = pg_pool.SimpleConnectionPool(
                    minconn=1, maxconn=10,
                    host=db_host,
                    port=int(db_port),
                    dbname=dbname,
                    user=unquote(db_user),
                    password=unquote(db_pass_enc),
                )
            else:
                # key=value DSN format (e.g. "host=... port=... dbname=... user=... password=...")
                # psycopg2 accepts this natively and handles special characters without encoding.
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

_MEETING_DESTINATION_RE = re.compile(
    r'\b(?:meeting|appointment|event|gathering|lunch|dinner|interview|client\s+meeting)\s+(?:in|at)\s+([A-Za-z][A-Za-z\s]+?)(?=\s+(?:next|this|on|for|with|about|tomorrow|today)|[,.]|$)',
    re.IGNORECASE,
)
_GOING_TO_RE = re.compile(
    r'\b(?:going|headed|traveling|travelling)\s+to\s+([A-Za-z][A-Za-z\s]+?)(?=\s+(?:for|next|this|on|tomorrow|today)|[,.]|$)',
    re.IGNORECASE,
)
_WILL_BE_RE = re.compile(
    r'\bwill\s+be\s+(?:in|at)\s+([A-Za-z][A-Za-z\s]+?)(?=\s+(?:for|next|this|on|tomorrow|today)|[,.]|$)',
    re.IGNORECASE,
)

def _extract_meeting_destination(text: str) -> str | None:
    """Extract named destination from meeting/travel-intent phrases. Returns title-cased name or None."""
    for pattern in (_MEETING_DESTINATION_RE, _GOING_TO_RE, _WILL_BE_RE):
        m = pattern.search(text)
        if m:
            return m.group(1).strip().title()
    return None


def process_persona(user_input: str):
    """Detect [legal ai] persona tag. Returns (persona, cleaned_input, filtered_tools, addendum_override)."""
    persona = "auto"
    filtered_tools = None
    addendum_override = None

    if user_input.lower().startswith("[legal ai]"):
        persona = "legal"
        user_input = user_input[10:].strip()
        legal_whitelist = ["search_legal", "summarize_legal_case"]
        filtered_tools = [t for t in _context.all_fun_manifest if t["function"]["name"] in legal_whitelist]
        try:
            with open("resources/prompts/legal_prompt.txt", "r", encoding="utf-8") as f:
                addendum_override = f.read()
        except Exception as e:
            logging.error(f"Failed to load legal_prompt.txt: {e}")

    elif user_input.lower().startswith("[garment]"):
        persona = "garment"
        user_input = user_input[9:].strip()
        garment_whitelist = ["recommend_garments"]
        filtered_tools = [t for t in _context.all_fun_manifest if t["function"]["name"] in garment_whitelist]
        addendum_override = (
            "GARMENT ASSISTANT MODE\n\n"
            "You are a helpful personal stylist and fashion advisor. "
            "Use the recommend_garments function to fetch weather data and outfit recommendations. "
            "If the user's gender is not clear from context, ask for it before calling the function.\n\n"
            "IMPORTANT — always request 4 outfit sets (sets=4) unless the user explicitly asks for fewer.\n\n"
            "IMPORTANT — weather handling: If the message contains [FRONTEND_WEATHER:{...}], "
            "you MUST extract that JSON string exactly as-is and pass it as the weather_json parameter "
            "when calling recommend_garments. Do not modify, summarize, or omit it. "
            "Never show, repeat, or mention the [FRONTEND_WEATHER] annotation in your response to the user — it is internal data only.\n\n"
            "RESPONSE LENGTH — Your chat message must be exactly 1 sentence. "
            "Just hand off to the cards — e.g. 'Here are your outfit picks.' "
            "No weather context, no vibe names, no summaries, no greetings."
        )

    elif user_input.lower().startswith("[maps]"):
        persona = "maps"
        user_input = user_input[6:].strip()
        maps_whitelist = ["search_nearby_places"]
        filtered_tools = [t for t in _context.all_fun_manifest if t["function"]["name"] in maps_whitelist]
        addendum_override = (
            "MAPS ASSISTANT MODE\n\n"
            "You are a helpful local guide and places discovery assistant. "
            "Use the search_nearby_places function to find places.\n\n"

            "LOCATION — the message may contain one of two annotations:\n"
            "• [MEETING_LOCATION:X] — the user is going TO place X for a meeting or event. "
            "You MUST call search_nearby_places with location_name=X. "
            "Use query='meeting place cafe restaurant hotel' unless the user's message gives more context. "
            "Use radius=10000 for provinces/regions, radius=5000 for cities.\n"
            "• [USER_LOCATION:{lat,lng}] — the user's current GPS position. "
            "Extract lat/lng and pass them to search_nearby_places. "
            "Never mention this annotation in your response.\n\n"

            "MULTI-LOCATION — if the user mentions multiple named places "
            "(e.g. 'SM Baguio, La Union, Vigan City'), pass ALL as a SINGLE comma-separated "
            "location_name in ONE call. Do NOT make separate calls per location.\n\n"

            "RESPONSE — when results are found, output NO prose. "
            "The frontend renders place cards. "
            "Only write one short sentence if zero results were returned."
        )

    elif user_input.lower().startswith("[stylist]"):
        persona = "stylist"
        user_input = user_input[9:].strip()
        stylist_whitelist = ["get_outfits_by_category", "recommend_garments", "get_cosmetics_by_skin_type", "recommend_cosmetics", "search_nearby_places", "scan_cosmetic", "match_cosmetics", "generate_outfit_image"]
        filtered_tools = [t for t in _context.all_fun_manifest if t["function"]["name"] in stylist_whitelist]
        addendum_override = (
            "You are Miraj, a personal AI stylist for the Mirror app. "
            "Mirror + mirage — you help people see possibilities in how they present themselves.\n\n"
            "Your scope: fashion, outfits, beauty, skincare, and self-expression. "
            "You are warm, direct, and opinionated. You have taste and you share it. "
            "You don't hedge — when you like something, say so. When something doesn't fit, redirect with care. "
            "Keep conversational responses to 1–2 sentences unless the user asks for more.\n\n"
            "TOOL USE — use the appropriate tool based on context and user intent:\n\n"
            "- OUTFIT REQUESTS — this is the most important rule: "
            "ANY message that names a clothing category, style, or occasion (e.g. 'Casual', 'Formal', 'Business', 'Streetwear', 'Athleisure', 'Sportswear', 'Vintage', 'Minimalist', or any similar word) "
            "MUST trigger an immediate call to get_outfits_by_category. "
            "Do NOT respond with text advice, outfit guides, or questions about what the user wants. "
            "Do NOT ask for clarification. Just call the tool. "
            "Pass the category value exactly as received — do not rephrase or translate it. "
            "Gender rule: "
            "1. If the user explicitly states a gender in the CURRENT message, use that. "
            "2. Otherwise use the [USER_GENDER:X] annotation — it is always the authoritative gender for this session. "
            "3. If no annotation and no gender stated, ask ONLY for gender — nothing else — then call the tool immediately once answered. "
            "Never read gender from conversation history.\n"
            "After the tool completes, respond with exactly 1 warm sentence — do not list items in text.\n\n"
            "- SKINCARE / COSMETICS REQUESTS — this is the most important rule: "
            "ANY message that names a skin type (e.g. 'Dry', 'Oily', 'Combination', 'Normal', 'Sensitive') "
            "in the context of skincare or beauty MUST trigger an immediate call to get_cosmetics_by_skin_type. "
            "Do NOT respond with text advice or ask for clarification. Just call the tool. "
            "Pass the skin type value exactly as received — do not rephrase or translate it. "
            "If [SKIN_TYPE:X] annotation is present, use that value. "
            "After the tool completes, respond with exactly 1 warm sentence — do not list products in text.\n\n"
            "- If the user's intent involves skincare or beauty but no skin type is stated or annotated: "
            "ask ONLY for their skin type — nothing else — then call get_cosmetics_by_skin_type immediately once answered.\n\n"
            "- If the user uploads or references a cosmetic product photo (front and back labels are in S3): "
            "call scan_cosmetic with the S3 keys and the user's skin type. "
            "After the tool completes, summarise the key ingredients and any concerns in 1–2 sentences.\n\n"
            "- If the user wants to know whether two scanned products are safe to use together: "
            "call match_cosmetics with the ingredient lists from the prior scans and the user's skin type. "
            "After the tool completes, give a clear verdict in 1–2 sentences.\n\n"
            "- If [USER_LOCATION:{...}] is present and the user wants to find nearby places: "
            "call search_nearby_places. Extract lat/lng from the JSON and pass them to the function. "
            "Respond with 1–2 sentences summarising what was found. "
            "Do NOT list places in text — the frontend renders place cards.\n\n"
            "- If the user explicitly wants to go somewhere in the app (not an outfit, cosmetics, or places request): "
            "call navigate_app. Pass target_url as the EXACT URL from this table — do not invent URLs. "
            "If nothing matches, tell the user instead.\n\n"
            "  /ai-assistant               → home / chat / start over\n"
            "  /ai-recommendation-fashion  → outfits & garment recommendations\n"
            "  /ai-recommendation-cosmetic → cosmetics & skincare\n"
            "  /map                        → map & nearby places\n"
            "  /overview                   → itinerary overview\n\n"
            "After the tool completes, respond with exactly 1 brief acknowledgment.\n\n"
            "- If [GARMENT_IMAGES:[...]] is present and the user's intent is to generate an outfit image: "
            "call generate_outfit_image. Extract the URL array exactly from the [GARMENT_IMAGES:...] annotation and pass as garment_image_urls. "
            "Use gender from [USER_GENDER:X] — apply the same gender rule as for get_outfits_by_category. "
            "After the tool completes, respond with exactly 1 warm sentence.\n\n"
            "NEVER show, repeat, or mention any annotation ([FRONTEND_WEATHER:...], [USER_LOCATION:...], "
            "[SKIN_ANALYSIS:...], [SITEMAP_CONTEXT:...], [USER_GENDER:...], [GARMENT_IMAGES:...]) in your response — all annotations are internal data only."
        )

    elif user_input.lower().startswith("[nav]"):
        persona = "nav"
        user_input = user_input[5:].strip()
        filtered_tools = []
        addendum_override = (
            "WAYFINDER MODE\n\n"
            "You are Wayfinder, a concise navigation assistant. Your only job is to map the user's query to a URL path.\n\n"
            "SITEMAP: The message contains [SITEMAP_CONTEXT:[...]] — a JSON array of valid URL paths. "
            "Extract it. These are the ONLY valid destinations. Never invent paths not in this list.\n\n"
            "OUTPUT: Respond with a single raw JSON object and nothing else. "
            "No markdown, no code fences, no explanation before or after the JSON.\n\n"
            "SCHEMA:\n"
            "{\n"
            '  "target_url": string | null,\n'
            '  "confidence": number,\n'
            '  "extracted_entities": {"name": string, "category": string} | null,\n'
            '  "system_message": string\n'
            "}\n\n"
            "RULES:\n"
            "- target_url: the best matching path from the sitemap, or null if no match exists.\n"
            "- confidence: 0.0 to 1.0. How certain you are this path satisfies the query.\n"
            "- extracted_entities: if the query names a specific person (name) or product/content type (category), extract it. Otherwise null.\n"
            "- system_message: one short sentence in Wayfinder's voice confirming the action or declining politely. No filler.\n\n"
            "USE CASES:\n"
            "1. Exact/semantic match — 'Take me to shoes' → target_url: /products/shoes, confidence: ~0.95\n"
            "2. Vague intent — 'Help with my order' → best semantic match like /orders or /support/faq, confidence: ~0.7\n"
            "3. Out of scope — 'What's the weather?' → target_url: null, confidence: 0.0, polite decline in system_message\n"
            "4. Dynamic param — 'Go to John's profile' → target_url: /profile or /profile/:id, extracted_entities: {\"name\": \"John\"}, confidence: ~0.9\n\n"
            "Never show, repeat, or mention the [SITEMAP_CONTEXT] annotation in your system_message — it is internal data only."
        )

    elif user_input.lower().startswith("[cosmetics]"):
        persona = "cosmetics"
        user_input = user_input[11:].strip()
        cosmetics_whitelist = ["recommend_cosmetics"]
        filtered_tools = [t for t in _context.all_fun_manifest if t["function"]["name"] in cosmetics_whitelist]
        addendum_override = (
            "COSMETICS ASSISTANT MODE\n\n"
            "You are a helpful skincare advisor and dermatology assistant. "
            "Use the recommend_cosmetics function to fetch personalised skincare routine recommendations.\n\n"
            "IMPORTANT — skin analysis handling: If the message contains [SKIN_ANALYSIS:{...}], "
            "you MUST extract that JSON string exactly as-is and pass it as the skin_analysis_json parameter "
            "when calling recommend_cosmetics. Do not modify, summarize, or omit it. "
            "Never show, repeat, or mention the [SKIN_ANALYSIS] annotation in your response to the user — it is internal data only.\n\n"
            "IMPORTANT — weather handling: If the message contains [FRONTEND_WEATHER:{...}], "
            "you MUST extract that JSON string exactly as-is and pass it as the weather_json parameter "
            "when calling recommend_cosmetics. Do not modify, summarize, or omit it. "
            "Never show, repeat, or mention the [FRONTEND_WEATHER] annotation in your response to the user — it is internal data only.\n\n"
            "IMPORTANT — location handling: If the message contains [USER_LOCATION:{...}], "
            "you MUST extract that JSON string exactly as-is and pass it as the location_json parameter "
            "when calling recommend_cosmetics. Do not modify, summarize, or omit it. "
            "Never show, repeat, or mention the [USER_LOCATION] annotation in your response to the user — it is internal data only.\n\n"
            "When presenting results, format each routine exactly like this:\n"
            "## Routine 1 — [Vibe Name]\n"
            "*[concern_note]*\n\n"
            "For each product in the routine, write its name in bold followed by the reason:\n"
            "**[Product Name]** (brand) — [reason]\n\n"
            "Do NOT include image tags or image URLs in your text response — images are handled separately by the frontend.\n\n"
            "Repeat the ## Routine N — [Vibe] header for each additional routine.\n\n"
            "RESPONSE LENGTH — Your opening chat message before the routines must be 3–4 sentences. "
            "Briefly summarise what the skin analysis reveals, how the local weather and location shaped these picks, and invite them to explore the results. "
            "Keep the tone warm and supportive. Do not repeat or list the products in text — the cards handle that."
        )

    elif user_input.lower().startswith("[tailor]"):
        persona = "tailor"
        user_input = user_input[8:].strip()
        tailor_whitelist = ["generate_outfit_image"]
        filtered_tools = [t for t in _context.all_fun_manifest if t["function"]["name"] in tailor_whitelist]
        addendum_override = (
            "TAILOR MODE\n\n"
            "You are a fashion tailor AI. Generate a ghost mannequin outfit photograph from the selected garment images.\n\n"
            "TOOL USE — call generate_outfit_image immediately. "
            "Extract garment_image_urls from [GARMENT_IMAGES:[...]] in the message. "
            "Extract gender from [GENDER:MALE] or [GENDER:FEMALE] in the message.\n\n"
            "Never show, mention, or repeat any annotation in your response — all annotations are internal data only.\n\n"
            "RESPONSE — after the tool completes, respond with exactly 1 short sentence confirming the outfit has been generated."
        )

    return persona, user_input, filtered_tools, addendum_override

# ---------------------------------------------------------------------------
# Legal RAG persona flow
# ---------------------------------------------------------------------------

def _generate_structured_data(legal_response: str, state) -> dict | None:
    """Second lightweight LLM call to produce TIMELINE and MINDMAP from the completed legal analysis."""
    try:
        prompt = (
            "You are a legal UI data generator. Based on the legal analysis below, "
            "return ONLY a JSON object with two keys: 'timeline' and 'mindMap'.\n\n"
            "timeline: array of 3–6 concrete legal steps the user should take.\n"
            "Each item: {\"title\": str, \"description\": str, \"status\": \"pending\", \"requires_previous\": bool}\n\n"
            "mindMap: tree rooted at the core legal issue.\n"
            "Shape: {\"id\": \"root\", \"label\": str, \"isRoot\": true, \"children\": [{\"id\": str, \"label\": str, \"children\": [...]}]}\n"
            "First-level children: Legal Basis, Key Facts, Remedies, Risks, Next Steps. Labels ≤ 6 words.\n\n"
            f"Legal analysis (first 2500 chars):\n{legal_response[:2500]}"
        )
        t0 = time.time()
        completion = state.openai_client.chat.completions.create(
            model=_context.model,
            messages=[
                {"role": "system", "content": "Return only valid JSON. No markdown, no explanation."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
        )
        raw = completion.choices[0].message.content or ""
        logging.info("_generate_structured_data %.2fs", time.time() - t0)
        return json.loads(raw)
    except Exception as e:
        logging.warning("_generate_structured_data failed: %s", e)
        return None


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

def _search_results_to_source_metadata(results: list) -> list:
    return [
        {
            "type": "legal_document",
            "item_id": str(r.get("item_id") or "").strip() or None,
            "title": r.get("title"),
            "category": r.get("metadata", {}).get("category") or r.get("type"),
            "bucket_slug": r.get("metadata", {}).get("bucket_slug"),
            "year": r.get("metadata", {}).get("year"),
            "source_url": r.get("url") or r.get("metadata", {}).get("source_url"),
            "s3_json_path": r.get("metadata", {}).get("s3_json_path"),
            "snippet": r.get("snippet"),
            "full_text": r.get("text"),
            "relevance": r.get("score", 1.0),
        }
        for r in results
    ]


def _resolve_citation_url(url: str) -> str:
    """Rewrite /sources/{id} → {LEGAL_LIBRARY_URL}/{id}. Pass other URLs through unchanged."""
    if url.startswith("/sources/"):
        doc_id = url[len("/sources/"):]
        base = _context.legal_library_url or ""
        return f"{base}/{doc_id}"
    return url


def format_legal_citation_links(text: str) -> str:
    if not text or not isinstance(text, str):
        return text
    url_pattern = r"(https?://[^)]+|[^)]+)"
    def repl_law(m):
        href = _resolve_citation_url(m.group(2))
        return f'<a href="{href}" class="legal-ref law" target="_blank">{m.group(1)}</a>'
    def repl_juris(m):
        href = _resolve_citation_url(m.group(2))
        return f'<a href="{href}" class="legal-ref jurisprudence" target="_blank">{m.group(1)}</a>'
    text = re.sub(r"\[([^\]]+ Law)\]\(" + url_pattern + r"\)", repl_law, text)
    text = re.sub(r"\[([^\]]+ Jurisprudence)\]\(" + url_pattern + r"\)", repl_juris, text)
    return text

def repair_legal_source_links(text: str, search_results) -> str:
    """Ensure /sources/... links map to ids present in current search results.

    Any missing/placeholder/out-of-set id is rewritten using the active result ids.
    """
    if not text or not isinstance(text, str) or not search_results:
        return text
    source_ids = []
    for r in search_results:
        raw = str(r.get("item_id") or r.get("id") or "").strip()
        if raw.isdigit():
            source_ids.append(raw)
    if not source_ids:
        return text

    # Keep only ids that currently exist in the legal documents table.
    # This protects against stale/mismatched ids leaking into /sources/{id}.
    try:
        legal_db = legal_rag_get_cached_db()
        if legal_db is not None:
            with legal_db.connect() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT id FROM documents WHERE id = ANY(%s::bigint[])",
                        ([int(x) for x in source_ids],),
                    )
                    existing = {str(row[0]) for row in cur.fetchall()}
            source_ids = [x for x in source_ids if x in existing]
    except Exception as e:
        logging.warning("[legal-citation] source id existence validation skipped: %s", e)

    if not source_ids:
        logging.warning("[legal-citation] no valid existing source ids available for repair")
        return text

    # Match /sources/{segment} in markdown links.
    # Keep numeric ids only when they exist in current search results.
    broken = re.compile(r"/sources/([^)\s\"\]]*)")
    idx = [0]
    source_id_set = set(source_ids)

    def repl(m):
        segment = m.group(1)
        if segment.isdigit() and segment in source_id_set:
            return m.group(0)
        replacement = source_ids[min(idx[0], len(source_ids) - 1)]
        idx[0] += 1
        return f"/sources/{replacement}"
    repaired = broken.sub(repl, text)

    # Hard validation gate: no outbound citation id may fall outside current search results.
    invalid_ids = []
    for m in broken.finditer(repaired):
        segment = m.group(1)
        if not (segment.isdigit() and segment in source_id_set):
            invalid_ids.append(segment)

    if invalid_ids:
        increment_metric_counter(
            "legal.citation_invalid_detected.count",
            value=len(invalid_ids),
            tags={"reason": "invalid_or_out_of_set_id"},
            session_id=None,
        )
        logging.warning(
            "[legal-citation] invalid source ids after repair=%s; forcing fallback id=%s",
            invalid_ids,
            source_ids[0],
        )
        increment_metric_counter(
            "legal.citation_repair.count",
            value=1,
            tags={"reason": "invalid_or_out_of_set_id"},
            session_id=None,
        )
        tracer = globals().get("broadcast_trace")
        if callable(tracer):
            try:
                tracer(
                    "action",
                    f"Legal citation guard rewrote invalid source ids: {invalid_ids} -> {source_ids[0]}",
                    None,
                )
            except Exception:
                # Trace must never break response generation.
                pass
        repaired = broken.sub(f"/sources/{source_ids[0]}", repaired)

    return repaired

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

    if bool(_context.fun_names):
        system_content += (
            "\n\n[XAI Transparency Requirement]\n"
            "When you decide to call a tool, write EXACTLY ONE LINE immediately before the tool call:\n"
            "REASON: <one sentence — why this tool, referencing the user's specific request>\n"
            "Write ONLY the REASON line. No other text before or after it when calling a tool.\n"
            "Never write a REASON line in a text-only response.\n"
            "The REASON line is for system logging only — do not reference it in your answer.\n"
        )

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
        if func_name == "navigate_app" and session_id:
            func_args["session_id"] = session_id
        result = globals()[func_name](**func_args)
        if func_name == "search_legal" and session_id and isinstance(result, dict):
            state = _context.sessions.get(session_id)
            if state is not None:
                state.last_search_legal_results = result.get("results", [])
        if func_name == "get_outfits_by_category" and session_id and isinstance(result, dict):
            state = _context.sessions.get(session_id)
            if state is not None:
                state.last_garment_result = result
        if func_name in ("recommend_cosmetics", "get_cosmetics_by_skin_type") and session_id and isinstance(result, dict):
            state = _context.sessions.get(session_id)
            if state is not None:
                state.last_cosmetics_result = result
            if func_name == "get_cosmetics_by_skin_type":
                _bg_skin_type = func_args.get("skin_type", "")
                if _bg_skin_type:
                    _SKIN_OILINESS = {"DRY": 20, "OILY": 80, "COMBINATION": 45, "NORMAL": 60, "SENSITIVE": 60}
                    _oiliness = _SKIN_OILINESS.get(_bg_skin_type.upper(), 50)
                    _skin_json = json.dumps({"output": [{"type": "oiliness", "ui_score": _oiliness}]})
                    def _scl_cosmetics(skin_json=_skin_json, skin_type=_bg_skin_type):
                        try:
                            r = globals()["recommend_cosmetics"](skin_analysis_json=skin_json, sets=1)
                            logging.info(f"[SCL_TRACE] recommend_cosmetics skin_type={skin_type} success={r.get('success')}")
                        except Exception as _e:
                            logging.warning(f"[SCL_TRACE] recommend_cosmetics failed: {_e}")
                    Thread(target=_scl_cosmetics, daemon=True).start()
        if func_name == "search_nearby_places" and session_id and isinstance(result, dict):
            state = _context.sessions.get(session_id)
            if state is not None:
                if result.get("multi_location"):
                    state.last_maps_result.extend(result.get("results", []))
                else:
                    state.last_maps_result.append(result)
        if func_name == "navigate_app" and session_id and isinstance(result, dict):
            state = _context.sessions.get(session_id)
            if state is not None:
                state.last_nav_result = result
        if func_name == "generate_outfit_image" and session_id and isinstance(result, dict):
            state = _context.sessions.get(session_id)
            if state is not None:
                state.last_tailor_result = result
        logging.info(f"Function {func_name} executed successfully.")
        return result
    except Exception as e:
        logging.error(f"Function {func_name} failed: {e}")
        return {"error": str(e)}

# ---------------------------------------------------------------------------
# Reason loop (non-streaming, collects full response)
# ---------------------------------------------------------------------------

def run_function_chain(state, messages: list, max_chains: int = 7, session_id: str = None, tools: list = None, query: str = ""):
    available_manifest = tools if tools is not None else _context.fun_manifest
    funcall_chains = []
    function_outputs = []
    full_response = ""
    last_tool = None

    def perform_chat(msgs):
        args = {
            "model": _context.model,
            "messages": msgs,
            "n": 1,
            "stream": True,
            "temperature": _context.temperature,
        }
        if available_manifest:
            args["tools"] = available_manifest
            args["tool_choice"] = "auto"
        return state.openai_client.chat.completions.create(**args)

    for _ in range(max_chains):
        function_call = {"name": None, "arguments": ""}
        if last_tool:
            _cycle_summary = f"The AI received results from '{last_tool}' and is deciding whether it has enough information to answer or needs to take another step."
        else:
            _cycle_summary = "The AI is working through the question, deciding whether it needs to use a tool or can answer directly."
        broadcast_trace("cognition", f"Cycle {_ + 1} — reasoning over {len(messages)} messages (model: {_context.model})", session_id,
            summary=_cycle_summary)
        stream_resp = perform_chat(messages)
        last_response = ""

        for chunk in stream_resp:
            delta = chunk.choices[0].delta
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                tc = delta.tool_calls[0]
                if tc.function.name:
                    function_call["name"] = tc.function.name
                if tc.function.arguments:
                    function_call["arguments"] += tc.function.arguments
            elif hasattr(delta, "function_call") and delta.function_call:
                fc = delta.function_call
                if fc.name:
                    function_call["name"] = fc.name
                if fc.arguments:
                    function_call["arguments"] += fc.arguments
            elif hasattr(delta, "content") and delta.content:
                last_response += delta.content.replace("~", "-")

        if not function_call["name"] and last_response:
            preview = last_response[:200].replace('\n', ' ')
            broadcast_trace("cognition", f"LLM produced final text ({len(last_response)} chars): \"{preview}{'…' if len(last_response) > 200 else ''}\"", session_id,
                summary="The AI has finished reasoning and is ready to deliver its response.")

        _xai_reason = None
        if last_response:
            _clean_lines = []
            for _ln in last_response.split("\n"):
                if _ln.strip().startswith("REASON:") and _xai_reason is None:
                    _xai_reason = _ln.strip()[7:].strip()
                else:
                    _clean_lines.append(_ln)
            last_response = "\n".join(_clean_lines).strip()

        if _xai_reason and function_call["name"]:
            broadcast_trace("cognition", f"Reasoning: {_xai_reason}", session_id,
                summary=f"In the AI's own words, it explained its decision: \"{_xai_reason}\"")

        if function_call["name"]:
            _tool_desc = next((t['function'].get('description', '') for t in available_manifest if t['function']['name'] == function_call['name']), '')
            _why_lines = [f"Proposed tool call: `{function_call['name']}`"]
            if _tool_desc:
                _why_lines.append(f"Why this tool: \"{_tool_desc[:200]}\"")
            try:
                _why_lines.append(f"Arguments passed: {json.dumps(json.loads(function_call['arguments']), ensure_ascii=False)}")
            except Exception:
                _why_lines.append(f"Arguments passed: {function_call['arguments'][:200]}")
            _args_desc = _describe_tool_args(function_call["name"], function_call["arguments"])
            _tool_summary = f"The AI decided it needs to use '{function_call['name']}' to answer this question. {_tool_desc}"
            if _args_desc:
                _tool_summary += f" {_args_desc}"
            broadcast_trace("cognition", "\n".join(_why_lines), session_id, summary=_tool_summary)

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
            broadcast_trace("control", f"BLOCKED duplicate call: `{function_call['name']}` — injecting memory reminder", session_id,
                summary=f"Safety check: The AI proposed to use '{function_call['name']}' again with the same inputs. This was blocked to prevent a redundant loop.")
            messages.append({"role": "system", "content": f"Function `{function_call['name']}` already called with same args. Do not repeat."})
            continue

        broadcast_trace("control", f"APPROVED: `{function_call['name']}` — no prior identical call found", session_id,
            summary=f"Safety check passed. The AI's proposed action is new — it has not taken this exact step before. Proceeding to execute '{function_call['name']}'.")
        broadcast_trace("action", f"Executing `{function_call['name']}`...", session_id,
            summary=f"The AI is now running '{function_call['name']}' to retrieve the information it needs.")

        funcall_chains.append({"name": function_call["name"], "args": cur_args})
        result = execute_function_call(function_call, session_id=session_id)
        if result is None:
            continue
        function_outputs.append((function_call["name"], result))
        last_tool = function_call["name"]
        state.turn_tool_calls = getattr(state, "turn_tool_calls", 0) + 1

        try:
            _rp = json.dumps(result, ensure_ascii=False)
        except Exception:
            _rp = str(result)
        _ctx = _summarize_tool_result(function_call["name"], result)
        broadcast_trace("action", f"Result from `{function_call['name']}`:\n{_rp[:300]}", session_id,
            summary=f"'{function_call['name']}' completed. {_ctx}")
        broadcast_trace("memory", f"Fact stored: `{function_call['name']}` result is now confirmed knowledge.\nValue: {_rp[:150]}", session_id,
            summary=f"The AI stored the result from '{function_call['name']}'. This confirmed knowledge will be used when composing the final response.")

        try:
            content = json.dumps(result, ensure_ascii=False)
        except Exception:
            content = str(result)
        messages.append({
            "role": "system",
            "content": (
                f"[Memory Fact]\nFunction `{function_call['name']}` returned:\n{content}\n\n"
                "Use this fact in all future reasoning. Do NOT re-call with the exact same arguments."
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

def _display_query(user_input: str) -> str:
    if user_input.startswith("[FRONTEND_WEATHER:"):
        parts = user_input.split("\n\n", 1)
        return parts[1].strip() if len(parts) > 1 else user_input
    return user_input


def _describe_input(text: str) -> str:
    bullet_count = sum(1 for line in text.splitlines() if line.strip().startswith("- "))
    is_structured = bullet_count >= 3 and (
        "rules:" in text.lower()
        or "intent decision" in text.lower()
        or text.strip().startswith("You are")
    )
    if is_structured:
        first_line = text.splitlines()[0].strip()
        return f"[Structured configuration]\n{first_line}\n{bullet_count} behavioral rule(s) defined."
    return text


def _interpret_score(score: float) -> str:
    if score >= 0.7:
        return "high"
    if score >= 0.4:
        return "moderate"
    return "low"


def _summarize_tool_result(tool_name: str, result) -> str:
    try:
        if tool_name == "search_legal":
            results = result.get("results", []) if isinstance(result, dict) else []
            if results:
                return f"The search returned {len(results)} result(s). The top match was: \"{results[0].get('title', 'unknown')[:80]}\""
            return "The search returned no results."
        if tool_name == "summarize_legal_case":
            title = (result.get("title") or result.get("case_title") or "unknown case") if isinstance(result, dict) else "unknown"
            return f"A case summary was produced for: \"{str(title)[:80]}\""
        if tool_name == "recommend_garments":
            recs = result.get("recommendations", []) if isinstance(result, dict) else []
            return f"{len(recs)} outfit recommendation(s) were returned."
        if tool_name == "search_nearby_places":
            places = result.get("places", []) if isinstance(result, dict) else []
            query = result.get("query", "") if isinstance(result, dict) else ""
            return f"{len(places)} place(s) found for '{query}'."
        if tool_name == "get_cosmetics_by_skin_type":
            skin_type = result.get("skin_type", "unknown") if isinstance(result, dict) else "unknown"
            return f"Cosmetics fetched for {skin_type} skin type."
        if tool_name == "recommend_cosmetics":
            skin_type = result.get("skin_type", "unknown") if isinstance(result, dict) else "unknown"
            concerns = result.get("concerns", []) if isinstance(result, dict) else []
            sets_req = result.get("sets_requested", 1) if isinstance(result, dict) else 1
            concern_str = f" Concerns flagged: {', '.join(concerns[:3])}." if concerns else ""
            return f"{sets_req} skincare routine(s) were generated for {skin_type} skin.{concern_str}"
        if tool_name == "get_legal_recommendation":
            issue = (result.get("issue") or "")[:80] if isinstance(result, dict) else ""
            mats = len(result.get("relevant_materials", [])) if isinstance(result, dict) else 0
            mat_str = f" {mats} material(s) referenced." if mats else ""
            return f'A legal recommendation was produced for: "{issue}".{mat_str}'
        if tool_name == "generate_legal_document":
            doc_name = (result.get("document_name") or result.get("document_type") or "document") if isinstance(result, dict) else "document"
            return f"A {doc_name} was drafted successfully."
        if tool_name == "analyze_document":
            fname = (result.get("filename") or "document") if isinstance(result, dict) else "document"
            chars = result.get("char_count", 0) if isinstance(result, dict) else 0
            trunc = " (truncated)" if isinstance(result, dict) and result.get("truncated") else ""
            return f'Analysis complete for "{fname}" ({chars:,} chars{trunc}).'
        if tool_name == "scan_cosmetic":
            product = (result.get("product_name") or result.get("product_title") or "unknown product") if isinstance(result, dict) else "unknown product"
            return f'Scan complete for "{str(product)[:60]}".'
        if tool_name == "match_cosmetics":
            verdict = (result.get("verdict") or "unknown") if isinstance(result, dict) else "unknown"
            reason = (result.get("verdict_reason") or "") if isinstance(result, dict) else ""
            return f"Compatibility verdict: {verdict}. {reason}".rstrip()
        if tool_name == "navigate_app":
            url = result.get("target_url") if isinstance(result, dict) else None
            conf = result.get("confidence", 0) if isinstance(result, dict) else 0
            if url:
                return f"Navigation resolved to: {url} (confidence: {int(conf * 100)}%)."
            return "No matching screen found for the navigation query."
        if tool_name == "generate_outfit_image":
            count = result.get("garment_count", "?") if isinstance(result, dict) else "?"
            gender = (result.get("gender") or "MALE") if isinstance(result, dict) else "MALE"
            return f"Outfit image generated for {gender} from {count} garment(s)."
        if tool_name == "execute_code":
            status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
            res_str = str(result.get("result", ""))[:100] if isinstance(result, dict) else ""
            if status == "success":
                return f"Code executed successfully. Result: {res_str}"
            return f"Code execution failed: {res_str}"
    except Exception:
        pass
    return "The tool completed and returned a result."


def _describe_tool_args(tool_name: str, arguments: str) -> str:
    try:
        args = json.loads(arguments) if isinstance(arguments, str) else arguments
        if tool_name == "search_legal":
            q = args.get("query", "")
            return f'It will search for: "{q}"' if q else ""
        if tool_name == "summarize_legal_case":
            item_id = args.get("item_id", "") or args.get("case_identifier", "")
            return f"It will retrieve document #{item_id}." if item_id else ""
        if tool_name == "recommend_garments":
            gender = args.get("gender", "")
            event_type = args.get("event_type", "")
            location = args.get("location", "")
            sets = args.get("sets", "")
            event_date = args.get("event_date", "")
            parts = []
            if gender: parts.append(gender)
            if event_type: parts.append(f"attending {event_type}")
            if location: parts.append(f"in {location}")
            if event_date: parts.append(f"on {event_date}")
            base = f"It will recommend {sets} outfit set(s)" if sets else "It will recommend outfit sets"
            return f"{base} for a {', '.join(parts)}." if parts else f"{base}."
        if tool_name == "search_nearby_places":
            query = args.get("query", "")
            location = args.get("location_name") or f"{args.get('lat', '')},{args.get('lng', '')}"
            radius = args.get("radius", 1500)
            return f'It will search for "{query}" near {location} (radius: {radius}m).'
        if tool_name == "get_cosmetics_by_skin_type":
            skin_type = args.get("skin_type", "unknown")
            return f"It will fetch cosmetic products for {skin_type} skin type."
        if tool_name == "recommend_cosmetics":
            sets = args.get("sets", 1)
            return f"It will generate {sets} skincare routine(s) based on the skin analysis."
        if tool_name == "get_legal_recommendation":
            issue = args.get("legal_issue", "")
            return f'It will research and provide a recommendation on: "{issue[:100]}".' if issue else ""
        if tool_name == "generate_legal_document":
            doc_type = args.get("document_type", "document")
            return f"It will draft a {doc_type}."
        if tool_name == "analyze_document":
            s3_key = args.get("s3_key", "")
            fname = args.get("filename") or (s3_key.split("/")[-1] if s3_key else "")
            return f'It will analyze the document: "{fname}".' if fname else ""
        if tool_name == "scan_cosmetic":
            skin_type = args.get("skin_type", "general")
            return f"It will scan a cosmetic product label for {skin_type} skin."
        if tool_name == "match_cosmetics":
            a = args.get("product_a_name", "product A")
            b = args.get("product_b_name", "product B")
            return f'It will check if "{a}" and "{b}" are compatible.'
        if tool_name == "navigate_app":
            query = args.get("query", "")
            return f'It will navigate to the screen matching: "{query[:80]}".' if query else ""
        if tool_name == "generate_outfit_image":
            urls = args.get("garment_image_urls", [])
            gender = args.get("gender", "MALE")
            return f"It will generate a {gender} outfit image from {len(urls)} garment image(s)."
        if tool_name == "execute_code":
            code = args.get("code", "")
            snippet = code[:80] + ("…" if len(code) > 80 else "")
            return f"It will execute: {snippet}" if code else ""
    except Exception:
        pass
    return ""


def _broadcast_retrieval_context(state, tools, addendum_override, session_id, query: str = "", persona: str = "auto"):
    rag_sources = getattr(state, "source_metadata", [])
    if rag_sources:
        lines = [f"RAG retrieved {len(rag_sources)} chunk(s) — injected as evidence:"]
        for s in rag_sources[:5]:
            score = f"{s.get('relevance', 0):.2f}" if s.get('relevance') is not None else "?"
            title = (s.get('title') or '')[:60]
            excerpt = (s.get('text_content') or '')[:150].replace('\n', ' ').strip()
            lines.append(f"  [{score}] {title}")
            if excerpt:
                lines.append(f"         \"{excerpt}{'…' if len(s.get('text_content','')) > 150 else ''}\"")
        top = rag_sources[0]
        top_title = (top.get('title') or '')[:80]
        top_score = top.get('relevance', 0) or 0
        broadcast_trace("retrieval", "\n".join(lines), session_id,
            summary=(
                f"The AI checked its knowledge base and found {len(rag_sources)} relevant source(s). "
                f"The top match, \"{top_title}\", had {_interpret_score(top_score)} relevance. "
                f"These documents will be used to ground the response in verified material."
            ))
    else:
        if persona in ("garment", "maps"):
            _no_rag_summary = "No knowledge base search was performed — this persona uses live external API data directly."
        else:
            _no_rag_summary = (
                "The AI checked its knowledge base but found no documents above the relevance threshold. "
                "It will answer using its training knowledge and the conversation history."
            )
        broadcast_trace("retrieval", "RAG not used — LLM relied solely on its training knowledge and conversation history", session_id,
            summary=_no_rag_summary)
    _persona_label = {"legal": "Legal AI", "garment": "Garment Stylist", "cosmetics": "Cosmetics Advisor", "maps": "Maps Guide", "auto": "General Assistant"}.get(persona, persona.title())
    available_tools = tools if tools is not None else _context.fun_manifest
    history_turns = len(state.prompt) if state.prompt else 0
    _history_desc = f"{history_turns} prior message(s)" if history_turns > 0 else "no prior context"
    _rag_context = (
        f"and was given {len(rag_sources)} document(s) from the knowledge base as context"
        if rag_sources else "with no documents from the knowledge base"
    )
    if available_tools:
        tool_lines = [f"Mode: {_persona_label} | History turns: {history_turns} | LLM was given {len(available_tools)} tool(s):"]
        for t in available_tools:
            fn = t['function']
            tool_lines.append(f"  • {fn['name']}: {fn.get('description', '')[:120]}")
        broadcast_trace("cognition", "\n".join(tool_lines), session_id,
            summary=(
                f"The AI is preparing to reason in {_persona_label} mode. "
                f"It is reviewing {_history_desc}, has {len(available_tools)} tool(s) available, {_rag_context}."
            ))
    else:
        broadcast_trace("cognition", f"Mode: {_persona_label} | History turns: {history_turns} | No tools available", session_id,
            summary=(
                f"The AI is preparing to reason in {_persona_label} mode with no tools. "
                f"It will answer directly from its training knowledge, reviewing {_history_desc}. {_rag_context.capitalize()}."
            ))

def _broadcast_turn_confidence(state, session_id):
    rag_count = len(getattr(state, "source_metadata", []))
    tool_count = getattr(state, "turn_tool_calls", 0)
    if rag_count > 0 and tool_count > 0:
        summary = f"This response is grounded in {rag_count} verified document(s) and used {tool_count} tool call(s)."
    elif rag_count > 0:
        summary = f"This response is grounded in {rag_count} verified document(s) from the knowledge base."
    elif tool_count > 0:
        summary = f"This response used {tool_count} tool call(s). No knowledge base documents were consulted."
    else:
        summary = "This response was generated entirely from the AI's training knowledge — no external sources were consulted."
    broadcast_trace("cognition", f"Turn complete — RAG: {rag_count} source(s), tools: {tool_count} call(s)", session_id, summary=summary)


def reason_loop(state, query: str, session_id: str = None, tools: list = None, addendum_override: str = None, persona: str = "auto"):
    messages = prepare_chat_messages(state, query, addendum_override=addendum_override)
    _broadcast_retrieval_context(state, tools, addendum_override, session_id, query=query, persona=persona)
    state.turn_tool_calls = 0
    result = run_function_chain(state, messages, session_id=session_id, tools=tools, query=query)
    _broadcast_turn_confidence(state, session_id)
    return result

# ---------------------------------------------------------------------------
# Streaming reason loop (generator)
# ---------------------------------------------------------------------------

async def _astream_llm(perform_chat_fn, messages):
    loop = asyncio.get_event_loop()
    q = asyncio.Queue()

    def _run():
        try:
            for chunk in perform_chat_fn(messages):
                loop.call_soon_threadsafe(q.put_nowait, chunk)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    Thread(target=_run, daemon=True).start()

    while True:
        item = await q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def streaming_run_function_chain(state, messages: list, max_chains: int = 7, session_id: str = None, tools: list = None, query: str = ""):
    available_manifest = tools if tools is not None else _context.fun_manifest
    funcall_chains = []
    function_outputs = []
    full_response = ""
    last_tool = None
    _chain_start = time.time()

    def perform_chat(msgs):
        args = {
            "model": _context.model,
            "messages": msgs,
            "n": 1,
            "stream": True,
            "temperature": _context.temperature,
        }
        if available_manifest:
            args["tools"] = available_manifest
            args["tool_choice"] = "auto"
        return state.openai_client.chat.completions.create(**args)

    for iteration in range(max_chains):
        function_call = {"name": None, "arguments": ""}
        if last_tool:
            _cycle_summary = f"The AI received results from '{last_tool}' and is deciding whether it has enough information to answer or needs to take another step."
        else:
            _cycle_summary = "The AI is working through the question, deciding whether it needs to use a tool or can answer directly."
        broadcast_trace("cognition", f"Cycle {iteration + 1} — reasoning over {len(messages)} messages (model: {_context.model})", session_id,
            summary=_cycle_summary)
        await asyncio.sleep(0)
        _xai_buffer = ""
        _xai_first_line_done = False
        _xai_reason = None
        _iter_start = time.time()
        last_response = ""
        _first_token_time = None

        async for chunk in _astream_llm(perform_chat, messages):
            delta = chunk.choices[0].delta
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                tc = delta.tool_calls[0]
                if tc.function.name:
                    function_call["name"] = tc.function.name
                if tc.function.arguments:
                    function_call["arguments"] += tc.function.arguments
            elif hasattr(delta, "function_call") and delta.function_call:
                fc = delta.function_call
                if fc.name:
                    function_call["name"] = fc.name
                if fc.arguments:
                    function_call["arguments"] += fc.arguments
            elif hasattr(delta, "content") and delta.content:
                if _first_token_time is None:
                    _first_token_time = time.time()
                part = delta.content.replace("~", "-")
                if not _xai_first_line_done:
                    _xai_buffer += part
                    if "\n" in _xai_buffer:
                        _xai_first_line_done = True
                        newline_pos = _xai_buffer.index("\n")
                        first_line = _xai_buffer[:newline_pos]
                        remainder = _xai_buffer[newline_pos + 1:]
                        if first_line.strip().startswith("REASON:"):
                            _xai_reason = first_line.strip()[7:].strip()
                        else:
                            last_response += first_line + "\n"
                            yield first_line + "\n"
                        if remainder:
                            last_response += remainder
                            yield remainder
                        _xai_buffer = ""
                else:
                    last_response += part
                    yield part

        if _xai_buffer and not _xai_first_line_done:
            if _xai_buffer.strip().startswith("REASON:"):
                _xai_reason = _xai_buffer.strip()[7:].strip()
            else:
                last_response += _xai_buffer
                yield _xai_buffer

        _iter_elapsed = time.time() - _iter_start
        if function_call["name"]:
            logging.info(
                "chain[%d] LLM→tool=%s llm=%.2fs session=%s",
                iteration, function_call["name"], _iter_elapsed, session_id,
            )
        else:
            ttft_iter = (_first_token_time - _iter_start) if _first_token_time else 0
            logging.info(
                "chain[%d] LLM→text chars=%d ttft=%.2fs total=%.2fs session=%s",
                iteration, len(last_response), ttft_iter, _iter_elapsed, session_id,
            )

        if not function_call["name"] and last_response:
            preview = last_response[:200].replace('\n', ' ')
            broadcast_trace("cognition", f"LLM produced final text ({len(last_response)} chars): \"{preview}{'…' if len(last_response) > 200 else ''}\"", session_id,
                summary="The AI has finished reasoning and is ready to deliver its response.")
            await asyncio.sleep(0)

        if _xai_reason and function_call["name"]:
            broadcast_trace("cognition", f"Reasoning: {_xai_reason}", session_id,
                summary=f"In the AI's own words, it explained its decision: \"{_xai_reason}\"")
            await asyncio.sleep(0)

        if function_call["name"]:
            _tool_desc = next((t['function'].get('description', '') for t in available_manifest if t['function']['name'] == function_call['name']), '')
            _why_lines = [f"Proposed tool call: `{function_call['name']}`"]
            if _tool_desc:
                _why_lines.append(f"Why this tool: \"{_tool_desc[:200]}\"")
            try:
                _why_lines.append(f"Arguments passed: {json.dumps(json.loads(function_call['arguments']), ensure_ascii=False)}")
            except Exception:
                _why_lines.append(f"Arguments passed: {function_call['arguments'][:200]}")
            _args_desc = _describe_tool_args(function_call["name"], function_call["arguments"])
            _tool_summary = f"The AI decided it needs to use '{function_call['name']}' to answer this question. {_tool_desc}"
            if _args_desc:
                _tool_summary += f" {_args_desc}"
            broadcast_trace("cognition", "\n".join(_why_lines), session_id, summary=_tool_summary)
            await asyncio.sleep(0)

        if last_response.strip():
            full_response = last_response.strip()

        if not function_call["name"]:
            break

        # HITL gate: emit pending_approval event and stop streaming
        if not _context.auto_approval:
            yield f"__HITL__{json.dumps({'function_call': function_call, 'messages': messages, 'tools': [t['function']['name'] for t in (tools or [])]})}"
            return

        try:
            cur_args = json.loads(function_call["arguments"])
        except Exception:
            cur_args = function_call["arguments"]

        is_dup = any(fc["name"] == function_call["name"] and fc["args"] == cur_args for fc in funcall_chains)
        if is_dup:
            broadcast_trace("control", f"BLOCKED duplicate call: `{function_call['name']}` — injecting memory reminder", session_id,
                summary=f"Safety check: The AI proposed to use '{function_call['name']}' again with the same inputs. This was blocked to prevent a redundant loop.")
            await asyncio.sleep(0)
            messages.append({"role": "system", "content": f"Function `{function_call['name']}` already called. Do not repeat."})
            continue

        broadcast_trace("control", f"APPROVED: `{function_call['name']}` — no prior identical call found", session_id,
            summary=f"Safety check passed. The AI's proposed action is new — it has not taken this exact step before. Proceeding to execute '{function_call['name']}'.")
        await asyncio.sleep(0)
        broadcast_trace("action", f"Executing `{function_call['name']}`...", session_id,
            summary=f"The AI is now running '{function_call['name']}' to retrieve the information it needs.")
        await asyncio.sleep(0)

        funcall_chains.append({"name": function_call["name"], "args": cur_args})

        _tool_start = time.time()
        result = await asyncio.to_thread(execute_function_call, function_call, session_id=session_id)
        logging.info(
            "chain[%d] tool=%s exec=%.2fs session=%s",
            iteration, function_call["name"], time.time() - _tool_start, session_id,
        )
        if result is None:
            continue
        function_outputs.append((function_call["name"], result))
        last_tool = function_call["name"]
        state.turn_tool_calls = getattr(state, "turn_tool_calls", 0) + 1

        try:
            _rp = json.dumps(result, ensure_ascii=False)
        except Exception:
            _rp = str(result)
        _ctx = _summarize_tool_result(function_call["name"], result)
        broadcast_trace("action", f"Result from `{function_call['name']}`:\n{_rp[:300]}", session_id,
            summary=f"'{function_call['name']}' completed. {_ctx}")
        await asyncio.sleep(0)
        broadcast_trace("memory", f"Fact stored: `{function_call['name']}` result is now confirmed knowledge.\nValue: {_rp[:150]}", session_id,
            summary=f"The AI stored the result from '{function_call['name']}'. This confirmed knowledge will be used when composing the final response.")
        await asyncio.sleep(0)

        try:
            content = json.dumps(result, ensure_ascii=False)
        except Exception:
            content = str(result)
        messages.append({
            "role": "system",
            "content": f"[Memory Fact]\nFunction `{function_call['name']}` returned:\n{content}\n\nUse this fact. Do NOT re-call with the exact same arguments.",
        })
        messages.append({
            "role": "system",
            "content": "[Constraints]\nIf a complete response has been produced, TERMINATE.",
        })

async def streaming_reason_loop(state, query: str, session_id: str = None, tools: list = None, addendum_override: str = None, persona: str = "auto"):
    messages = prepare_chat_messages(state, query, addendum_override=addendum_override)
    _broadcast_retrieval_context(state, tools, addendum_override, session_id, query=query, persona=persona)
    await asyncio.sleep(0)
    state.turn_tool_calls = 0
    async for chunk in streaming_run_function_chain(state, messages, session_id=session_id, tools=tools, query=query):
        yield chunk
    _broadcast_turn_confidence(state, session_id)
    await asyncio.sleep(0)

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health_check():
    """Check connectivity to all dependent services. Returns 200 if healthy, 503 if any critical service is down."""
    import time
    checks = {}
    overall = "healthy"

    # PostgreSQL — prefer the LegalDatabase pool (used by /chat and /chat-stream legal persona),
    # fall back to the startup pool (_context.db_pool) if legal services not yet initialized.
    try:
        legal_db = legal_rag_get_cached_db()
        if legal_db is not None:
            t0 = time.monotonic()
            with legal_db.connect() as conn:
                conn.cursor().execute("SELECT 1")
            checks["postgres"] = {"status": "ok", "latency_ms": round((time.monotonic() - t0) * 1000)}
        elif _context.db_pool:
            t0 = time.monotonic()
            conn = _context.db_pool.getconn()
            try:
                conn.cursor().execute("SELECT 1")
            finally:
                _context.db_pool.putconn(conn)
            checks["postgres"] = {"status": "ok", "latency_ms": round((time.monotonic() - t0) * 1000)}
        else:
            checks["postgres"] = {"status": "unconfigured"}
    except Exception as e:
        checks["postgres"] = {"status": "error", "detail": str(e)}
        overall = "degraded"

    # S3 legal bucket
    try:
        legal_bucket = os.getenv("LEGAL_S3_BUCKET_NAME")
        if not legal_bucket:
            checks["s3_legal"] = {"status": "unconfigured"}
        else:
            t0 = time.monotonic()
            s3 = s3_storage.get_s3_client()
            s3.head_bucket(Bucket=legal_bucket)
            checks["s3_legal"] = {"status": "ok", "bucket": legal_bucket, "latency_ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:
        checks["s3_legal"] = {"status": "error", "bucket": os.getenv("LEGAL_S3_BUCKET_NAME"), "detail": str(e)}
        overall = "degraded"

    # S3 cosmetics bucket
    try:
        cosmetics_bucket = os.getenv("COSMETICS_S3_BUCKET_NAME")
        cosmetics_region = os.getenv("COSMETICS_AWS_REGION")
        if not cosmetics_bucket:
            checks["s3_cosmetics"] = {"status": "unconfigured"}
        else:
            t0 = time.monotonic()
            s3 = s3_storage.get_s3_client(bucket_name=cosmetics_bucket, region=cosmetics_region)
            s3.head_bucket(Bucket=cosmetics_bucket)
            checks["s3_cosmetics"] = {"status": "ok", "bucket": cosmetics_bucket, "latency_ms": round((time.monotonic() - t0) * 1000)}
    except Exception as e:
        checks["s3_cosmetics"] = {"status": "error", "bucket": os.getenv("COSMETICS_S3_BUCKET_NAME"), "detail": str(e)}
        overall = "degraded"

    # OpenAI
    if _context.openai_api_key:
        checks["openai"] = {"status": "configured", "model": _context.model}
    else:
        checks["openai"] = {"status": "unconfigured"}
        overall = "degraded"

    status_code = 200 if overall == "healthy" else 503
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status_code,
        content={"status": overall, "checks": checks},
    )


@app.get("/config")
async def get_config():
    """Public config for frontend clients — no secrets."""
    return {"legal_library_url": _context.legal_library_url}


@app.get("/version")
async def get_version():
    """Runtime build marker for deployment verification."""
    return {
        "service": "chat-wonder-v2-api",
        "build_marker": _context.build_marker,
        "chat_model": _context.model,
        "citation_guard": {
            "enabled": True,
            "strict_out_of_set_validation": True,
            "metrics": [
                "legal.citation_invalid_detected.count",
                "legal.citation_repair.count",
            ],
        },
    }


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

    if persona == "garment" and request.weather:
        try:
            user_input = f"[FRONTEND_WEATHER:{json.dumps(request.weather, ensure_ascii=False)}]\n\n{user_input}"
        except Exception:
            pass

    if persona == "cosmetics":
        if request.skin_analysis:
            try:
                user_input = f"[SKIN_ANALYSIS:{json.dumps(request.skin_analysis, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass
        if request.weather:
            try:
                user_input = f"[FRONTEND_WEATHER:{json.dumps(request.weather, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass
        if request.location:
            try:
                user_input = f"[USER_LOCATION:{json.dumps(request.location, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass

    if persona == "maps":
        _meeting_dest = _extract_meeting_destination(user_input)
        if _meeting_dest:
            user_input = f"[MEETING_LOCATION:{_meeting_dest}]\n\n{user_input}"
        elif request.location:
            try:
                user_input = f"[USER_LOCATION:{json.dumps(request.location, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass
        if session_id and session_id in _context.sessions:
            _context.sessions[session_id].last_maps_result = []

    if persona == "nav" and request.sitemap_context:
        try:
            user_input = f"[SITEMAP_CONTEXT:{json.dumps(request.sitemap_context, ensure_ascii=False)}]\n\n{user_input}"
        except Exception:
            pass

    if persona == "stylist":
        if session_id and session_id in _context.sessions and request.sitemap_context:
            _context.sessions[session_id].sitemap_context = request.sitemap_context
        if request.weather:
            try:
                user_input = f"[FRONTEND_WEATHER:{json.dumps(request.weather, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass
        if request.location:
            try:
                _meeting_dest = _extract_meeting_destination(user_input)
                if _meeting_dest:
                    user_input = f"[MEETING_LOCATION:{_meeting_dest}]\n\n{user_input}"
                else:
                    user_input = f"[USER_LOCATION:{json.dumps(request.location, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass
            if session_id and session_id in _context.sessions:
                _context.sessions[session_id].last_maps_result = []
        if request.skin_analysis:
            try:
                user_input = f"[SKIN_ANALYSIS:{json.dumps(request.skin_analysis, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass
        if request.sitemap_context:
            try:
                user_input = f"[SITEMAP_CONTEXT:{json.dumps(request.sitemap_context, ensure_ascii=False)}]\n\n{user_input}"
            except Exception:
                pass

    if getattr(request, "document_context", None):
        doc_injection = (
            "\n\n[COSMETICS CATALOG — use ONLY if the user's current request is about skincare, "
            "beauty, or cosmetics products. If the user is asking about outfits, garments, or "
            "fashion, ignore this section entirely and do NOT call recommend_cosmetics.]\n"
            + request.document_context
        )
        addendum_override = (addendum_override or "You are a helpful assistant.") + doc_injection

    if not user_input.strip():
        raise HTTPException(status_code=400, detail="User input is empty.")
    if not session_id or session_id not in _context.sessions:
        raise HTTPException(status_code=401, detail="Unknown session.")

    state = _context.sessions[session_id]
    state.last_used = time.time()
    if request.user_id:
        state.user_id = request.user_id

    if not _context.openai_api_key:
        raise HTTPException(status_code=400, detail="API key is required.")
    init_openai_client(state, _context.openai_api_key)

    _tool_count = len(filtered_tools) if filtered_tools is not None else len(_context.fun_manifest)
    _persona_label = {"legal": "Legal AI", "garment": "Garment Stylist", "cosmetics": "Cosmetics Advisor", "maps": "Maps Guide", "nav": "Wayfinder", "stylist": "Miraj", "auto": "General Assistant"}.get(persona, persona.title())
    broadcast_trace("request", f"New turn — session {session_id} — input: {user_input[:120]}", session_id,
        summary=f"A new question was received.\n\nPersona: {_persona_label} — {_tool_count} tool(s) available.\n\n{_describe_input(_display_query(user_input))}")

    # Structured outfit search — bypass Miraj LLM when category is provided
    if persona == "stylist" and request.category:
        _meta = request.category.get("meta", "")
        _gender = request.gender or state.confirmed_gender or "MALE"
        _sets = max(1, min(4, request.sets or 3))
        _outfit_result = search_outfits_by_category(
            gender=_gender,
            meta_categories=_meta,
            location=request.location,
            weather=request.weather,
            sets=_sets,
        )
        if _outfit_result.get("success"):
            state.last_outfit_ids_result = _outfit_result.get("ids", [])
            if _outfit_result.get("gender", "").upper() in ("MALE", "FEMALE"):
                state.confirmed_gender = _outfit_result["gender"].upper()
            def _scl_outfit_search(gender=_gender, meta=_meta):
                try:
                    r = search_outfits_by_category(gender=gender, meta_categories=meta, sets=1)
                    logging.info(f"[SCL_TRACE] search_outfits_by_category gender={gender} meta={meta} success={r.get('success')}")
                except Exception as _e:
                    logging.warning(f"[SCL_TRACE] search_outfits_by_category failed: {_e}")
            Thread(target=_scl_outfit_search, daemon=True).start()
        else:
            logging.warning(f"[stylist] search_outfits_by_category failed: {_outfit_result.get('error')}")
        logging.info("/chat [stylist/direct] %.2fs session=%s", time.time() - _t_start, session_id)
        return {
            "response": "",
            "outfit_ids": state.last_outfit_ids_result,
            "nav_result": {"target_url": "/ai-recommendation-fashion", "confidence": 1.0, "extracted_entities": None, "system_message": ""},
        }

    # Normal path with optional HITL (legal, garment, cosmetics, maps, nav, stylist personas always auto-approve)
    _was_auto = _context.auto_approval
    if persona in ("legal", "garment", "cosmetics", "maps", "nav", "stylist"):
        _context.auto_approval = True
    try:
        result = reason_loop(state, user_input, session_id=session_id, tools=filtered_tools, addendum_override=addendum_override, persona=persona)
    finally:
        _context.auto_approval = _was_auto

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
    if persona == "legal" and state.last_search_legal_results:
        state.source_metadata = _search_results_to_source_metadata(state.last_search_legal_results)
    _do_nav_extract = persona == "nav"
    if _do_nav_extract:
        try:
            nav_json = json.loads(final_text)
            if nav_json.get("confidence", 0) < 0.5:
                nav_json["target_url"] = None
            state.last_nav_result = nav_json
            final_text = nav_json.get("system_message", "")
        except Exception:
            state.last_nav_result = {"target_url": None, "confidence": 0.0, "extracted_entities": None, "system_message": final_text}
    if persona == "stylist":
        if state.last_garment_result and not (state.last_nav_result or {}).get("target_url"):
            state.last_nav_result = {"target_url": "/ai-recommendation-fashion", "confidence": 1.0, "extracted_entities": None, "system_message": ""}
        elif state.last_cosmetics_result and not (state.last_nav_result or {}).get("target_url"):
            state.last_nav_result = {"target_url": "/ai-recommendation-cosmetic", "confidence": 1.0, "extracted_entities": None, "system_message": ""}
        elif state.last_maps_result and not (state.last_nav_result or {}).get("target_url"):
            state.last_nav_result = {"target_url": "/map", "confidence": 1.0, "extracted_entities": None, "system_message": ""}
    state.generated.append(final_text)
    _context.sessions[session_id] = state

    logging.info("/chat [%s] %.2fs session=%s", persona, time.time() - _t_start, session_id)
    return {
        "response": final_text,
        "lookup": state.lookup,
        "source_metadata": state.source_metadata,
        "garment_sets": state.last_garment_result if persona in ("garment", "stylist") and state.last_garment_result else None,
        "cosmetics_sets": state.last_cosmetics_result if persona in ("cosmetics", "stylist") and state.last_cosmetics_result else None,
        "places_results": state.last_maps_result if persona in ("maps", "stylist") and state.last_maps_result else None,
        "nav_result": state.last_nav_result if persona in ("nav", "stylist") and state.last_nav_result else None,
        "tailor_result": state.last_tailor_result if persona == "tailor" and state.last_tailor_result else None,
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
    _hitl_outcome = "approved" if decision == "approve" else "rejected"
    _hitl_next = "The AI will proceed." if decision == "approve" else "The AI will respond without taking this action."
    broadcast_trace("control", f"HITL decision: {decision} — tool: `{fc['name']}`", session_id,
        summary=f"A human reviewer {_hitl_outcome} the AI's proposed action. {_hitl_next}")
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
        _resume_query = _display_query(state.prompt[-1]) if state.prompt else ""
        cont_result = run_function_chain(state, messages, session_id=session_id, tools=available_manifest, query=_resume_query)
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
        "garment_sets": state.last_garment_result if state.last_garment_result else None,
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

                available_manifest = [t for t in _context.all_fun_manifest if t["function"]["name"] in (tools or [])] if tools else _context.fun_manifest
                _context.auto_approval = True
                full_response = ""
                try:
                    _resume_query = _display_query(state.prompt[-1]) if state.prompt else ""
                    async for chunk in streaming_run_function_chain(state, messages, session_id=session_id, tools=available_manifest, query=_resume_query):
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
            if data.get("user_id"):
                state.user_id = data["user_id"]
            init_openai_client(state, _context.openai_api_key)

            user_input = request.user_input or getattr(request, "user_history_select", "") or ""
            if not user_input.strip():
                await websocket.send_text("[Error] User input is empty.")
                await websocket.send_text(_context.__END__)
                continue

            persona, user_input, filtered_tools, addendum_override = process_persona(user_input)

            # Inject frontend-provided weather for garment persona
            if persona == "garment" and data.get("weather"):
                try:
                    user_input = f"[FRONTEND_WEATHER:{json.dumps(data['weather'], ensure_ascii=False)}]\n\n{user_input}"
                except Exception:
                    pass

            # Inject frontend-provided skin analysis, weather, and location for cosmetics persona
            if persona == "cosmetics":
                if data.get("skin_analysis"):
                    try:
                        user_input = f"[SKIN_ANALYSIS:{json.dumps(data['skin_analysis'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass
                if data.get("weather"):
                    try:
                        user_input = f"[FRONTEND_WEATHER:{json.dumps(data['weather'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass
                if data.get("location"):
                    try:
                        user_input = f"[USER_LOCATION:{json.dumps(data['location'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass

            # Inject frontend-provided location for maps persona
            if persona == "maps":
                _meeting_dest = _extract_meeting_destination(user_input)
                if _meeting_dest:
                    user_input = f"[MEETING_LOCATION:{_meeting_dest}]\n\n{user_input}"
                elif data.get("location"):
                    try:
                        user_input = f"[USER_LOCATION:{json.dumps(data['location'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass
                state.last_maps_result = []

            # Inject sitemap for nav persona (B2: runs sync, no streaming)
            if persona == "nav" and data.get("sitemap_context"):
                try:
                    user_input = f"[SITEMAP_CONTEXT:{json.dumps(data['sitemap_context'], ensure_ascii=False)}]\n\n{user_input}"
                except Exception:
                    pass

            if persona == "stylist":
                if data.get("sitemap_context"):
                    state.sitemap_context = data["sitemap_context"]
                if data.get("weather"):
                    try:
                        user_input = f"[FRONTEND_WEATHER:{json.dumps(data['weather'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass
                if data.get("location"):
                    try:
                        _meeting_dest = _extract_meeting_destination(user_input)
                        if _meeting_dest:
                            user_input = f"[MEETING_LOCATION:{_meeting_dest}]\n\n{user_input}"
                        else:
                            user_input = f"[USER_LOCATION:{json.dumps(data['location'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass
                    state.last_maps_result = []
                if data.get("skin_analysis"):
                    try:
                        user_input = f"[SKIN_ANALYSIS:{json.dumps(data['skin_analysis'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass
                try:
                    # Resolve gender: prefer client-sent DB value, fall back to server-side
                    # state (set immediately when recommend_garments returns, no round-trip).
                    _raw_gender = (data.get("gender") or "").strip().upper() or state.confirmed_gender
                    if _raw_gender in ("MALE", "FEMALE"):
                        user_input = f"[USER_GENDER:{_raw_gender}]\n\n{user_input}"
                except Exception:
                    pass
                if data.get("sitemap_context"):
                    try:
                        user_input = f"[SITEMAP_CONTEXT:{json.dumps(data['sitemap_context'], ensure_ascii=False)}]\n\n{user_input}"
                    except Exception:
                        pass

            if getattr(request, "document_context", None):
                doc_injection = (
                    "\n\n[COSMETICS CATALOG — use ONLY if the user's current request is about skincare, "
                    "beauty, or cosmetics products. If the user is asking about outfits, garments, or "
                    "fashion, ignore this section entirely and do NOT call recommend_cosmetics.]\n"
                    + request.document_context
                )
                addendum_override = (addendum_override or "You are a helpful assistant.") + doc_injection

            _tool_count = len(filtered_tools) if filtered_tools is not None else len(_context.fun_manifest)
            _persona_label = {"legal": "Legal AI", "garment": "Garment Stylist", "cosmetics": "Cosmetics Advisor", "maps": "Maps Guide", "nav": "Wayfinder", "stylist": "Miraj", "tailor": "Tailor", "auto": "General Assistant"}.get(persona, persona.title())
            broadcast_trace("request", f"New turn — session {session_id} — input: {user_input[:120]}", session_id,
                summary=f"A new question was received.\n\nPersona: {_persona_label} — {_tool_count} tool(s) available.\n\n{_describe_input(_display_query(user_input))}")

            full_response = ""
            _ws_t_start = time.time()
            _ws_t_first_chunk = None
            _was_auto = _context.auto_approval
            end_sent = False
            if persona in ("legal", "garment", "cosmetics", "maps", "nav", "stylist", "tailor"):
                _context.auto_approval = True

            # B2: nav runs sync — no streaming of raw JSON
            if persona == "nav":
                try:
                    nav_result_raw = reason_loop(state, user_input, session_id=session_id, tools=[], addendum_override=addendum_override, persona=persona)
                    nav_text = (nav_result_raw or "").strip()
                    try:
                        nav_json = json.loads(nav_text)
                        if nav_json.get("confidence", 0) < 0.5:
                            nav_json["target_url"] = None
                        state.last_nav_result = nav_json
                        system_message = nav_json.get("system_message", "")
                    except Exception:
                        nav_json = {"target_url": None, "confidence": 0.0, "extracted_entities": None, "system_message": nav_text}
                        state.last_nav_result = nav_json
                        system_message = nav_text
                    state.prompt.append(user_input)
                    state.generated.append(system_message)
                    _context.sessions[session_id] = state
                    await websocket.send_text(system_message)
                    await websocket.send_text(f"[NAV_DATA]{json.dumps(state.last_nav_result)}")
                    await websocket.send_text(_context.__END__)
                    await websocket.send_text("[DONE]")
                except Exception as e:
                    logging.warning("/chat-stream [nav] ERROR session=%s: %s", session_id, e)
                    await websocket.send_text("[Error] Navigation failed.")
                    await websocket.send_text(_context.__END__)
                    await websocket.send_text("[DONE]")
                finally:
                    _context.auto_approval = _was_auto
                continue

            # Structured outfit search — bypass streaming LLM when category is provided
            if persona == "stylist" and data.get("category"):
                _meta = data["category"].get("meta", "")
                _gender = (data.get("gender") or "").strip().upper() or state.confirmed_gender or "MALE"
                _sets = max(1, min(4, data.get("sets") or 3))
                _outfit_result = search_outfits_by_category(
                    gender=_gender,
                    meta_categories=_meta,
                    location=data.get("location"),
                    weather=data.get("weather"),
                    sets=_sets,
                )
                if _outfit_result.get("success"):
                    state.last_outfit_ids_result = _outfit_result.get("ids", [])
                    if _outfit_result.get("gender", "").upper() in ("MALE", "FEMALE"):
                        state.confirmed_gender = _outfit_result["gender"].upper()
                    def _scl_ws(gender=_gender, meta=_meta):
                        try:
                            r = search_outfits_by_category(gender=gender, meta_categories=meta, sets=1)
                            logging.info(f"[SCL_TRACE] search_outfits_by_category gender={gender} meta={meta} success={r.get('success')}")
                        except Exception as _e:
                            logging.warning(f"[SCL_TRACE] search_outfits_by_category failed: {_e}")
                    Thread(target=_scl_ws, daemon=True).start()
                else:
                    logging.warning(f"[stylist/ws] search_outfits_by_category failed: {_outfit_result.get('error')}")
                state.prompt.append(user_input)
                state.generated.append("")
                _context.sessions[session_id] = state
                await websocket.send_text(f"[OUTFIT_IDS]{json.dumps(state.last_outfit_ids_result)}")
                await websocket.send_text(_context.__END__)
                _context.auto_approval = _was_auto
                continue

            try:
                async for chunk in streaming_reason_loop(state, user_input, session_id=session_id, tools=filtered_tools, addendum_override=addendum_override, persona=persona):
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
                    if _ws_t_first_chunk is None:
                        _ws_t_first_chunk = time.time()
                    await websocket.send_text(chunk)
                    full_response += chunk

                if full_response:
                    state.prompt.append(user_input)
                    final_text = full_response.strip()
                    final_text = repair_legal_source_links(final_text, state.last_search_legal_results)
                    if addendum_override and "LEGAL ASSISTANT MODE" in addendum_override:
                        final_text = format_legal_citation_links(final_text)
                    if persona == "legal" and state.last_search_legal_results:
                        state.source_metadata = _search_results_to_source_metadata(state.last_search_legal_results)
                        await websocket.send_text(f"[Sources] {json.dumps(state.source_metadata)}")
                    state.generated.append(final_text)
                    _context.sessions[session_id] = state
                else:
                    # Tool was called but LLM produced no text — still persist the turn
                    state.prompt.append(user_input)
                    state.generated.append("")
                    _context.sessions[session_id] = state
                # Structured data frames fire regardless of whether LLM produced text,
                # so the panel renders even when the LLM terminates silently after a tool call.
                if persona == "stylist":
                    if state.last_tailor_result and not (state.last_nav_result or {}).get("target_url"):
                        state.last_nav_result = {"target_url": "/ai-recommendation-fashion", "confidence": 1.0, "extracted_entities": None, "system_message": ""}
                    elif state.last_garment_result and not (state.last_nav_result or {}).get("target_url"):
                        state.last_nav_result = {"target_url": "/ai-recommendation-fashion", "confidence": 1.0, "extracted_entities": None, "system_message": ""}
                    elif state.last_cosmetics_result and not (state.last_nav_result or {}).get("target_url"):
                        state.last_nav_result = {"target_url": "/ai-recommendation-cosmetic", "confidence": 1.0, "extracted_entities": None, "system_message": ""}
                    elif state.last_maps_result and not (state.last_nav_result or {}).get("target_url"):
                        state.last_nav_result = {"target_url": "/map", "confidence": 1.0, "extracted_entities": None, "system_message": ""}
                if persona in ("garment", "stylist") and state.last_garment_result:
                    await websocket.send_text(f"[GARMENT_DATA]{json.dumps(state.last_garment_result)}")
                _garment_gender = (state.last_garment_result or {}).get("gender", "").upper()
                if _garment_gender in ("MALE", "FEMALE"):
                    await websocket.send_text(f"[GENDER_UPDATE]{_garment_gender}")
                if persona in ("cosmetics", "stylist") and state.last_cosmetics_result:
                    await websocket.send_text(f"[COSMETICS_DATA]{json.dumps(state.last_cosmetics_result)}")
                if persona in ("maps", "stylist") and state.last_maps_result:
                    await websocket.send_text(f"[MAPS_DATA]{json.dumps(state.last_maps_result)}")
                if persona in ("tailor", "stylist") and state.last_tailor_result:
                    await websocket.send_text(f"[TAILOR_DATA]{json.dumps(state.last_tailor_result)}")
                # nav emission disabled for stylist — front end handles navigation
                _ws_t_end = time.time()
                ttft = (_ws_t_first_chunk - _ws_t_start) if _ws_t_first_chunk else 0
                logging.info(
                    "/chat-stream [%s] ttft=%.2fs total=%.2fs chars=%d session=%s",
                    persona, ttft, _ws_t_end - _ws_t_start, len(full_response), session_id,
                )
                # Send __END__ now so the client unlocks immediately, then generate
                # timeline/mindmap in a background thread and send before [DONE].
                await websocket.send_text(_context.__END__)
                if persona == "legal" and full_response:
                    t_sd = time.time()
                    structured = await asyncio.to_thread(_generate_structured_data, full_response.strip(), state)
                    logging.info("_generate_structured_data %.2fs", time.time() - t_sd)
                    if structured:
                        await websocket.send_text(f"[STRUCTURED_DATA]{json.dumps(structured)}")
                await websocket.send_text("[DONE]")
                end_sent = True

            except Exception as e:
                logging.warning(
                    "/chat-stream [%s] ERROR after %.2fs session=%s: %s",
                    persona, time.time() - _ws_t_start, session_id, e,
                )
                await websocket.send_text(f"[Error] {e}")
                await websocket.send_text(_context.__END__)
                end_sent = True
            finally:
                _context.auto_approval = _was_auto
                if not end_sent:
                    try:
                        await websocket.send_text(_context.__END__)
                    except Exception:
                        pass

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
            all_loaded = [ec_manifest] + [item for item in loaded if item.get("function", {}).get("name") != "execute_code"]
            _context.all_fun_manifest = all_loaded
            _context.fun_manifest = [t for t in all_loaded if not t.get("persona")]
            _context.fun_names = [t["function"]["name"] for t in _context.fun_manifest]
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

        session_id = getattr(request, "session_id", None)
        broadcast_trace("request", f"Legal search — query: {prompt[:120]}", session_id,
            summary=f"A search of the Philippine legal database was requested for: \"{prompt[:200]}\"")

        # Reuse optimized_query on page 2+ to avoid an extra GPT call
        if request.optimized_query:
            optimized_query = request.optimized_query.strip()
        else:
            broadcast_trace("action", "Optimizing query with LLM...", session_id,
                summary="The AI is refining the search query to improve accuracy.")
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

        broadcast_trace("action", f"Running pgvector search — optimized query: {optimized_query[:120]}", session_id,
            summary=f"Searching the legal database using the optimised query: \"{optimized_query[:200]}\"")
        rag_result = legal_rag_search(query=optimized_query, limit=_LEGAL_SEARCH_MAX_POOL)
        rag_rows = rag_result.get("results", []) if isinstance(rag_result, dict) else []
        broadcast_trace("retrieval", f"Legal search returned {len(rag_rows)} result(s)", session_id,
            summary=f"The legal database returned {len(rag_rows)} result(s) matching the query.")

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

        broadcast_trace("request", f"Legal case fetch — item: {item_id}", None,
            summary=f"Fetching full legal document with ID {item_id} from the database.")
        doc = legal_rag_get_document(int(item_id))
        if not isinstance(doc, dict):
            raise HTTPException(status_code=404, detail=f"Case '{item_id}' not found.")

        broadcast_trace("retrieval", f"Fetched legal document: {str(doc.get('title', ''))[:100]}", None,
            summary=f"Retrieved legal document: '{str(doc.get('title', ''))[:100]}'")
        metadata = doc.get("metadata_json") or {}
        return {
            "id": doc.get("id"),
            "item_id": str(doc.get("id")),
            "type": doc.get("category"),
            "title": doc.get("title"),
            "url": doc.get("source_url"),
            "text_content": doc.get("full_text") or doc.get("summary") or doc.get("concise_summary") or "",
            "formatted_markdown": doc.get("formatted_markdown"),
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


def _format_and_store_legal_markdown(
    document_id: int,
    force: bool = False,
    generate_title: bool = True,
) -> dict:
    db = legal_rag_get_cached_db()
    if db is None:
        from legal_rag.router import _services
        db = _services()[1]

    doc = db.get_document(document_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is required")

    existing_title = (doc.get("title") or "").strip() or None
    existing_md = (doc.get("formatted_markdown") or "").strip()
    if existing_md and not force and (existing_title or not generate_title):
        return {
            "item_id": str(document_id),
            "title": existing_title,
            "title_generated": False,
            "formatted_markdown": existing_md,
            "cached": True,
        }

    source_text = doc.get("full_text") or doc.get("summary") or doc.get("concise_summary") or ""
    if not str(source_text).strip():
        raise HTTPException(status_code=400, detail="Document has no text to format")

    model = os.getenv("LEGAL_CHAT_MODEL", "gpt-4o-mini")
    title, markdown, title_generated = format_document_combined(
        str(source_text),
        existing_title=existing_title,
        generate_title=generate_title,
        category=doc.get("category"),
        case_no=doc.get("case_no"),
        openai_api_key=api_key,
        model=model,
        openai_base_url=os.getenv("OPENAI_BASE_URL"),
    )
    if not markdown:
        raise HTTPException(status_code=500, detail="Formatter returned empty markdown")

    if title_generated and title:
        db.set_document_title(document_id, title)

    markdown = prepend_title_heading(markdown, title or existing_title)
    db.set_formatted_markdown(document_id, markdown)
    return {
        "item_id": str(document_id),
        "title": title or existing_title,
        "title_generated": title_generated,
        "formatted_markdown": markdown,
        "cached": False,
    }


@app.post("/api/legal/format-document/{item_id}")
async def api_format_legal_document(
    item_id: str,
    force: bool = Query(False),
    generate_title: bool = Query(True, description="Generate documents.title when empty"),
):
    """Generate structured markdown (and title when missing) for a legal document."""
    try:
        if not str(item_id).isdigit():
            raise HTTPException(status_code=400, detail="item_id must be a numeric legal document id.")
        broadcast_trace("request", f"Legal format — item: {item_id}", None,
            summary=f"Preparing to format legal document {item_id} into readable markdown.")
        broadcast_trace("action", "Calling LLM to format legal document as markdown...", None,
            summary="The AI is converting this legal document into clean, structured markdown.")
        result = _format_and_store_legal_markdown(int(item_id), force=force, generate_title=generate_title)
        broadcast_trace("memory", "Formatted markdown stored.", None,
            summary="The formatted version of this document has been saved to the database.")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[Legal Format Document] Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _list_document_ids_to_format(force: bool = False, limit: int | None = 50, all_docs: bool = False) -> list[int]:
    db = legal_rag_get_cached_db()
    if db is None:
        from legal_rag.router import _services
        db = _services()[1]

    limit_clause = "" if all_docs else "LIMIT %s"
    params: tuple = () if all_docs else (limit,)
    if force:
        sql = f"""
            SELECT id FROM documents
            WHERE full_text IS NOT NULL AND length(trim(full_text)) > 100
            ORDER BY id
            {limit_clause}
        """
    else:
        sql = f"""
            SELECT id FROM documents
            WHERE formatted_markdown IS NULL
              AND full_text IS NOT NULL
              AND length(trim(full_text)) > 100
            ORDER BY id
            {limit_clause}
        """
    with db.connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return [row[0] for row in cur.fetchall()]


@app.post("/api/legal/format-documents")
async def api_format_legal_documents(
    force: bool = Query(False, description="Reformat even when formatted_markdown already exists"),
    limit: int = Query(50, ge=1, le=5000, description="Max documents per request (ignored when all=true)"),
    all_docs: bool = Query(False, alias="all", description="Process every matching document"),
    delay: float = Query(0.5, ge=0, le=10, description="Seconds between OpenAI calls"),
    generate_title: bool = Query(True, description="Generate documents.title when empty"),
):
    """
    Batch-format documents into formatted_markdown.

    curl examples:
      curl -X POST 'http://localhost:8000/api/legal/format-documents?limit=10'
      curl -X POST 'http://localhost:8000/api/legal/format-documents?all=true'
      curl -X POST 'http://localhost:8000/api/legal/format-document/150'
    """
    try:
        doc_ids = _list_document_ids_to_format(force=force, limit=None if all_docs else limit, all_docs=all_docs)
        broadcast_trace("request", f"Legal format-documents — {len(doc_ids)} document(s) to format", None,
            summary=f"Batch formatting {len(doc_ids)} legal document(s) into readable markdown.")
        if not doc_ids:
            return {
                "total": 0,
                "ok": 0,
                "failed": 0,
                "cached": 0,
                "formatted": 0,
                "titles_generated": 0,
                "errors": [],
                "message": "No documents to format",
            }

        ok = 0
        failed = 0
        cached = 0
        formatted = 0
        titles_generated = 0
        errors: list[dict] = []

        for doc_id in doc_ids:
            try:
                result = _format_and_store_legal_markdown(
                    doc_id, force=force, generate_title=generate_title
                )
                ok += 1
                if result.get("cached"):
                    cached += 1
                else:
                    formatted += 1
                if result.get("title_generated"):
                    titles_generated += 1
            except HTTPException as exc:
                failed += 1
                errors.append({"item_id": str(doc_id), "detail": exc.detail})
            except Exception as exc:
                failed += 1
                errors.append({"item_id": str(doc_id), "detail": str(exc)})

            if delay > 0:
                time.sleep(delay)

        return {
            "total": len(doc_ids),
            "ok": ok,
            "failed": failed,
            "cached": cached,
            "formatted": formatted,
            "titles_generated": titles_generated,
            "errors": errors[:50],
        }
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[Legal Format Documents] Error: {e}")
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
    session_id = getattr(request, "session_id", None)
    broadcast_trace("request", f"Legal document analysis — file: {filename}", session_id,
        summary=f"Starting AI analysis of uploaded document: '{filename}'")

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
                broadcast_trace("action", "Transcribing audio with Whisper...", session_id,
                    summary="Converting audio to text using speech recognition.")
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
                broadcast_trace("action", "Running vision OCR on document image...", session_id,
                    summary="Extracting text from the document image using AI vision.")
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
        broadcast_trace("retrieval", f"Text extracted — {len(extracted_text)} chars from '{filename}'", session_id,
            summary=f"Successfully extracted {len(extracted_text)} characters of text from '{filename}'. The AI will now analyse the content.")

        ai_summary = None
        if _context.openai_api_key:
            try:
                broadcast_trace("action", "Analysing document content with LLM...", session_id,
                    summary="The AI is reading the document and producing a structured legal analysis.")
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
                broadcast_trace("cognition", f"Analysis complete — {len(ai_summary)} chars extracted", session_id,
                    summary=f"Legal analysis complete. The AI produced a {len(ai_summary)}-character structured report.")
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
    broadcast_trace("request", f"Upload and analyse — file: {filename}", None,
        summary=f"Document '{filename}' uploaded and queued for AI legal analysis.")
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

    session_id = getattr(request, "session_id", None)
    broadcast_trace("request", f"Legal synthesis — {len(request.summaries)} document(s)", session_id,
        summary=f"Starting cross-document synthesis across {len(request.summaries)} document(s).")

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

    broadcast_trace("action", "Calling LLM to synthesize across documents...", session_id,
        summary="The AI is reading all documents together and producing a unified legal analysis.")
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
        broadcast_trace("cognition", f"Synthesis complete — {len(synthesis)} chars", session_id,
            summary=f"Synthesis complete. The AI produced a {len(synthesis)}-character unified analysis across all documents.")
        logging.info(f"[Synthesize Documents] Synthesis generated for {len(request.summaries)} documents.")
        return {"success": True, "synthesis": synthesis}
    except Exception as e:
        logging.error(f"[Synthesize Documents] Error: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to generate synthesis: {e}")


# ---------------------------------------------------------------------------
# Cosmetics
# ---------------------------------------------------------------------------

@app.post("/api/cosmetics/scan")
async def scan_cosmetic_product(request: CosmeticScanRequest):
    """Analyze a cosmetic product using S3 keys for its front and back label images."""
    if not _context.openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    broadcast_trace("request", f"Cosmetics scan — s3_key: {request.back_s3_key[:60]}", request.session_id,
        summary="Starting ingredient analysis for the submitted cosmetic product.")
    from resources.functions.user_functions import scan_cosmetic
    broadcast_trace("action", "Running cosmetic ingredient scan...", request.session_id,
        summary="The AI is analysing the product's ingredient list.")
    analysis = scan_cosmetic(
        front_s3_key=request.front_s3_key or "",
        back_s3_key=request.back_s3_key,
        skin_type=request.skin_type or "general",
    )

    if not analysis.get("success"):
        raise HTTPException(status_code=500, detail=analysis.get("error", "Scan failed."))

    if request.session_id and request.session_id in _context.sessions:
        state = _context.sessions[request.session_id]
        product_name = analysis.get("product_name", "this product")
        seed_message = (
            f"I've scanned **{product_name}** for you. Here's the ingredient analysis:\n\n"
            f"{analysis.get('summary', '')}\n\n"
            "Feel free to ask me anything about the ingredients or whether this product is right for you."
        )
        state.generated.append({"role": "assistant", "content": seed_message})

    broadcast_trace("cognition", "Scan complete.", request.session_id,
        summary="Ingredient scan complete. Results are ready.")
    logging.info(f"[Cosmetics Scan] {analysis.get('product_name', 'unknown')} — front: {request.front_s3_key}")
    return analysis


@app.post("/api/cosmetics/match")
async def match_cosmetic_products(request: CosmeticMatchRequest):
    """Scan two cosmetic back-label images from S3 and check if their ingredients are compatible."""
    if not _context.openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    broadcast_trace("request", f"Cosmetics match — A: {request.product_a_s3_key[:40]} B: {request.product_b_s3_key[:40]}", request.session_id,
        summary="Checking compatibility between two cosmetic products.")
    from resources.functions.user_functions import scan_cosmetic, match_cosmetics

    skin_type = request.skin_type or "general"

    broadcast_trace("action", "Scanning both products...", request.session_id,
        summary="The AI is scanning both products before checking compatibility.")
    scan_a = scan_cosmetic(front_s3_key="", back_s3_key=request.product_a_s3_key, skin_type=skin_type)
    if not scan_a.get("success"):
        raise HTTPException(status_code=500, detail=f"Failed to scan product A: {scan_a.get('error')}")

    scan_b = scan_cosmetic(front_s3_key="", back_s3_key=request.product_b_s3_key, skin_type=skin_type)
    if not scan_b.get("success"):
        raise HTTPException(status_code=500, detail=f"Failed to scan product B: {scan_b.get('error')}")

    broadcast_trace("action", "Running compatibility match...", request.session_id,
        summary="The AI is comparing the ingredient profiles of both products.")
    result = match_cosmetics(
        product_a_name=scan_a.get("product_name") or request.product_a_s3_key,
        product_a_ingredients=scan_a.get("ingredients", []),
        product_b_name=scan_b.get("product_name") or request.product_b_s3_key,
        product_b_ingredients=scan_b.get("ingredients", []),
        skin_type=skin_type,
    )

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "Match failed."))

    result["product_a"] = {"name": scan_a.get("product_name"), "s3_key": request.product_a_s3_key}
    result["product_b"] = {"name": scan_b.get("product_name"), "s3_key": request.product_b_s3_key}

    if request.session_id and request.session_id in _context.sessions:
        state = _context.sessions[request.session_id]
        name_a = scan_a.get("product_name", "Product A")
        name_b = scan_b.get("product_name", "Product B")
        verdict = result.get("verdict", "unknown")
        seed_message = (
            f"I've checked the compatibility of **{name_a}** and **{name_b}**. "
            f"Verdict: **{verdict}**.\n\n{result.get('summary', '')}\n\n"
            "Feel free to ask me anything about using these products together."
        )
        state.generated.append({"role": "assistant", "content": seed_message})

    broadcast_trace("cognition", "Match complete.", request.session_id,
        summary="Compatibility check complete. Results are ready.")
    logging.info(f"[Cosmetics Match] {scan_a.get('product_name')} + {scan_b.get('product_name')} → {result.get('verdict')}")
    return result


@app.post("/api/tailor/generate")
async def tailor_generate_outfit(
    image: UploadFile = File(..., description="PNG canvas snapshot of the T-pose outfit"),
    gender: str = Form(..., description="MALE or FEMALE"),
):
    """
    REST endpoint for the Generate Outfit feature.
    Accepts a canvas blob (PNG) from OutfitPreviewCanvas.getBlob() plus the
    user's gender, calls gpt-image-1, uploads the result to S3, and returns
    a presigned GET URL. Separate from the WebSocket [tailor] persona which
    takes individual garment URLs.
    """
    if not _context.openai_api_key:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured.")

    gender_upper = (gender or "MALE").strip().upper()
    if gender_upper not in ("MALE", "FEMALE"):
        raise HTTPException(status_code=422, detail="gender must be MALE or FEMALE")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="image file is empty")

    import base64
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    gender_word = "female" if gender_upper == "FEMALE" else "male"

    prompt = (
        f"Using the provided garment images as the only reference, create a premium ghost mannequin "
        f"fashion product photograph combining all garments into one complete outfit. "
        f"Reproduce EVERY garment EXACTLY as shown — same color, same cut, same length, same fabric texture, "
        f"same prints, logos, patterns, stitching, proportions, wrinkles, folds, and construction details. "
        f"Do not substitute, redesign, replace, simplify, or generate alternative versions of any garment. "
        f"Reconstruct the outfit on a completely invisible {gender_word} mannequin with realistic body volume, "
        f"natural garment draping, and accurate layering. The mannequin must be entirely hidden — "
        f"no visible head, neck, face, arms, hands, wrists, legs, feet, ankles, skin, mannequin parts, or display stands. "
        f"Subtle luxury pose: one leg slightly forward, weight shifted to back leg, slight knee bend, "
        f"relaxed shoulders, clean editorial silhouette. "
        f"If shoes are present, they must appear naturally worn with no visible feet, ankles, or shoe interiors. "
        f"If a trouser or skirt hem falls above the shoe collar, the gap between hem and shoe must be empty white space — never skin, ankle, or leg. "
        f"If the hem reaches the shoe, it must meet the shoe collar with zero gap — no skin visible between hem and shoe. "
        f"If a bag is present, it must hang from the shoulder strap alone — no visible hand, wrist, or fingers gripping the strap or the bag body. "
        f"All accessories (belts, scarves, jewelry) must appear worn on the invisible mannequin with no body parts visible. "
        f"If any top is a crop top, tube top, or tank top, any exposed midriff, torso, or armhole area must show empty space or the garment's interior edge — never skin, flesh, or body. "
        f"Shoulder straps and armholes must not reveal shoulder skin, arm skin, or collarbone — the mannequin is entirely invisible beneath. "
        f"Any gap between garments (e.g., between a crop top hem and a waistband) must be empty space, not skin. "
        f"Studio-quality luxury e-commerce fashion photography. Centered full-body view. "
        f"Pure white seamless background (#FFFFFF). Ultra-sharp focus. Professional catalog lighting. "
        f"Negative: visible body parts, mannequin structure, display stand, floor reflection, "
        f"extra garments, duplicated clothing, watermark, cropped outfit, redesigned clothing, "
        f"visible hands, visible wrists, visible fingers, visible ankles, visible socks, skin at hem, skin at cuff, "
        f"shoe interior visible, trouser hem floating above shoe, leg between hem and shoe, ankle between hem and shoe, garment color changed, garment style changed, "
        f"hand gripping bag, hand on strap, wrist near bag, "
        f"visible midriff skin, visible torso skin, visible shoulder skin, visible collarbone, visible armhole skin, "
        f"skin between garments, flesh gap, exposed body between top and bottom."
    )

    from openai import OpenAI
    client = OpenAI(api_key=_context.openai_api_key)
    try:
        gen = client.images.edit(
            model="gpt-image-1",
            image=("outfit.png", image_bytes, "image/png"),
            prompt=prompt,
            n=1,
        )

        image_b64 = gen.data[0].b64_json if gen.data else None

        if not image_b64:
            raise HTTPException(status_code=500, detail="No image returned by gpt-image-1.")

        result_bytes = base64.b64decode(image_b64)

        tailor_bucket = os.getenv("TAILOR_S3_BUCKET_NAME")
        tailor_region = os.getenv("TAILOR_AWS_REGION")
        s3_key = f"tailor/generated/{uuid.uuid4()}.png"

        uploaded = s3_storage.upload_bytes_to_s3(
            result_bytes, s3_key, content_type="image/png",
            bucket_name=tailor_bucket, region=tailor_region,
        )
        if not uploaded:
            raise HTTPException(status_code=500, detail="Failed to upload generated outfit image to S3.")

        image_url = s3_storage.generate_presigned_get(
            s3_key, bucket_name=tailor_bucket, region=tailor_region,
        )
        if not image_url:
            raise HTTPException(status_code=500, detail="Failed to generate presigned URL.")

        logging.info(f"[tailor_generate_outfit] gender={gender_upper} s3_key={s3_key}")
        return {"success": True, "image_url": image_url, "s3_key": s3_key, "gender": gender_upper}

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"[tailor_generate_outfit] Failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/proxy-image")
async def proxy_image(url: str = Query(..., description="Public image URL to proxy")):
    """Fetches an external image server-side and returns it. Used by the test
    HTML so canvas.toBlob() works without CORS taint."""
    import urllib.request as _urlreq
    try:
        req = _urlreq.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with _urlreq.urlopen(req, timeout=15) as resp:
            data = resp.read()
            content_type = resp.headers.get("Content-Type", "image/png")
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Could not fetch image: {e}")
    return Response(content=data, media_type=content_type.split(";")[0].strip())
