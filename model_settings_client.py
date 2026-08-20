# -*- coding: utf-8 -*-
"""Client for ilovelawyer-api's live model-settings endpoint
(POST-migration path: GET /api/v1/model-settings).

Caches the full settings list for 5 minutes (user-confirmed interval) and falls
back to this app's own env var if the endpoint is unreachable or a tool isn't
listed there — the same fallback chain documented in
docs/plans/deepseek-provider-flexibility-plan.md.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.request
from typing import Dict, Optional

_CACHE_TTL_SECONDS = 300  # 5 minutes, user-confirmed
_REQUEST_TIMEOUT_SECONDS = 5

_logger = logging.getLogger(__name__)
_cache: Dict[str, object] = {"settings": None, "fetched_at": 0.0}


def _fetch_settings() -> Optional[Dict[str, str]]:
    base = os.getenv("ILOVELAWYER_API_BASE", "").rstrip("/")
    api_key = os.getenv("CHAT_WONDER_API_KEY", "")
    if not base:
        return None

    url = f"{base}/api/v1/model-settings"
    req = urllib.request.Request(url, headers={"x-api-key": api_key})
    try:
        with urllib.request.urlopen(req, timeout=_REQUEST_TIMEOUT_SECONDS) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        _logger.warning(
            "model_settings_client: fetch failed, falling back to .env for this call: %s", e
        )
        return None

    rows = data.get("settings", [])
    return {row["toolName"]: row["currentModel"] for row in rows if "toolName" in row}


def resolve_model(tool_name: str, env_var: str, default: str) -> str:
    """Resolve the model for `tool_name`: settings endpoint (cached) -> env var -> default.

    Does not handle a request-payload override — callers check that themselves
    before falling through to this function, per the plan's resolution order.
    """
    now = time.time()
    stale = _cache["settings"] is None or (now - float(_cache["fetched_at"])) > _CACHE_TTL_SECONDS
    if stale:
        fetched = _fetch_settings()
        if fetched is not None:
            _cache["settings"] = fetched
            _cache["fetched_at"] = now
        # fetch failed: keep whatever's already cached (even if stale) rather
        # than discarding it, so a transient outage doesn't force every call
        # back to the plain .env default while the cache still has a recent
        # real value.

    settings = _cache["settings"]
    if settings and tool_name in settings:
        return settings[tool_name]

    return os.getenv(env_var, default)
