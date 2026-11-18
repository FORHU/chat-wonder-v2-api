# -*- coding: utf-8 -*-
"""Persona routing overrides (CW-8, chat-wonder-v2-api#149): manual auto/manual model pinning,
stored and controlled in ilovelawyer-api/ilovelawyer-admin (FORHU/ilovelawyer-api#663,
FORHU/ilovelawyer-admin#45), not here -- chat-wonder-v2-api has no Postgres table or admin UI
of its own for this.

This module is a thin, read-only polling client: `get_override(persona)` is the only thing
model_router.py (CW-6/CW-7, not built yet) needs to call. It refreshes the whole table from
ilovelawyer-api's GET /api/persona-routing (api-key guarded, same CHAT_WONDER_API_KEY used
elsewhere) at most every ROUTING_OVERRIDE_CACHE_TTL_S seconds -- never per-turn, never
per-tool-call. A refresh failure keeps serving the last-known-good snapshot (fail open to the
last good state, not to an error); only a persona with no row in a snapshot that has at least
been populated once, or a server that has never once reached ilovelawyer-api, falls back to that
persona's env vars.
"""

import json
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

_logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_CACHE: Dict[str, dict] = {}
_LAST_REFRESHED_AT: Optional[float] = None  # time.monotonic(), None until the first success
_LAST_ERROR: Optional[str] = None


def _cache_ttl_s() -> float:
    try:
        return float(os.getenv("ROUTING_OVERRIDE_CACHE_TTL_S", "30"))
    except ValueError:
        return 30.0


def _is_stale(now: float) -> bool:
    if _LAST_REFRESHED_AT is None:
        return True
    # >=, not >: a TTL of 0 (tests, or an operator who wants every call to poll) must mean
    # "always stale", even when two calls land in the same monotonic tick.
    return now - _LAST_REFRESHED_AT >= _cache_ttl_s()


def _fetch_overrides() -> list:
    """One GET to ilovelawyer-api's chat-wonder-facing read route. Raises on any failure --
    callers decide what "failure" means (fail open to the last snapshot)."""
    base = (os.getenv("ILOVELAWYER_API_BASE") or "").rstrip("/")
    api_key = os.getenv("CHAT_WONDER_API_KEY", "")
    if not base:
        raise RuntimeError("ILOVELAWYER_API_BASE is not set")
    if not api_key:
        raise RuntimeError("CHAT_WONDER_API_KEY is not set")
    req = urllib.request.Request(f"{base}/api/persona-routing", headers={"x-api-key": api_key})
    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data.get("overrides") or []


def refresh(force: bool = False) -> bool:
    """Refreshes the cache now, regardless of TTL when `force` is set. Returns whether the
    refresh succeeded. On failure, the previous snapshot (if any) is left untouched."""
    global _LAST_REFRESHED_AT, _LAST_ERROR
    try:
        rows = _fetch_overrides()
    except Exception as e:  # noqa: BLE001 -- any failure here must never surface to the caller
        _LAST_ERROR = str(e)
        _logger.warning("persona routing override refresh failed, keeping last-known snapshot: %s", e)
        return False
    with _LOCK:
        _CACHE.clear()
        for row in rows:
            persona = row.get("persona")
            if persona:
                _CACHE[persona] = row
        _LAST_REFRESHED_AT = time.monotonic()
    _LAST_ERROR = None
    return True


def _ensure_fresh() -> None:
    now = time.monotonic()
    if _is_stale(now):
        refresh()


def _env_fallback(persona: str) -> dict:
    """A persona with no row in a reachable ilovelawyer-api, or no ilovelawyer-api ever reached
    at all -- same env vars the original design doc proposed, kept as the deploy-time escape
    hatch this never needs in the common case."""
    prefix = persona.upper()
    mode = (os.getenv(f"{prefix}_ROUTING_MODE") or "AUTO").upper()
    if mode not in ("AUTO", "MANUAL"):
        mode = "AUTO"
    manual_pinned_model = os.getenv(f"{prefix}_MANUAL_MODEL") or None
    raw_tool_overrides = os.getenv(f"{prefix}_TOOL_MODEL_OVERRIDES")
    tool_overrides = None
    if raw_tool_overrides:
        try:
            tool_overrides = json.loads(raw_tool_overrides)
        except ValueError:
            _logger.warning("%s_TOOL_MODEL_OVERRIDES is not valid JSON, ignoring", prefix)
    return {
        "persona": persona,
        "routingMode": mode,
        "manualPinnedModel": manual_pinned_model if mode == "MANUAL" else None,
        "toolOverrides": tool_overrides,
        "source": "env",
    }


def get_override(persona: str) -> dict:
    """The only function model_router.py needs. Always returns a usable dict -- `source` is
    "cache" when it came from ilovelawyer-api (even a stale snapshot still counts as "cache",
    since fail-open means we trust it until a fresher one arrives) or "env" when nothing has ever
    been cached for this persona."""
    _ensure_fresh()
    with _LOCK:
        row = _CACHE.get(persona)
    if row is not None:
        return {**row, "source": "cache"}
    return _env_fallback(persona)


def last_refresh_error() -> Optional[str]:
    """For a health/observability endpoint, not for the hot path."""
    return _LAST_ERROR


def _reset_for_tests() -> None:
    global _LAST_REFRESHED_AT, _LAST_ERROR
    with _LOCK:
        _CACHE.clear()
    _LAST_REFRESHED_AT = None
    _LAST_ERROR = None
