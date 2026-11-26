# -*- coding: utf-8 -*-
"""Glass-box trace broadcast and lightweight counter metrics (SCL — Supervised
Cognitive Loop). Extracted from the_server.py (chat-wonder-v2-api#182).

broadcast_trace needs each event's session's trace_turn_id, but has no reason
to own the session store itself — register_session_lookup lets the_server
hand it a lookup (`_context.sessions.get`) once at import time instead of
this module reaching for the server's global ApiContext. set_event_loop is
the same idea for the asyncio loop captured at startup.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Callable, Optional

_trace_queues: set = set()
_app_event_loop = None  # captured at startup so worker threads can schedule on the main loop
_metric_counters: dict = {}

_session_lookup: Optional[Callable[[str], object]] = None


def register_session_lookup(fn: Callable[[str], object]) -> None:
    global _session_lookup
    _session_lookup = fn


def set_event_loop(loop) -> None:
    global _app_event_loop
    _app_event_loop = loop


def broadcast_trace(event_type: str, text: str, session_id: str = None, summary: str = None):
    # Resolved here, not in a ContextVar: this runs from worker threads too, and the session
    # object already carries every other per-turn field (one turn at a time per session).
    _state = _session_lookup(session_id) if (_session_lookup and session_id) else None
    turn_id = getattr(_state, "trace_turn_id", None)
    data = json.dumps({"type": event_type, "text": text.strip(), "summary": summary.strip() if summary else None, "session_id": session_id, "turn_id": turn_id, "ts": time.time()})
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


def broadcast_model_decision(persona: str, decision_kind: str, from_model: str, to_model: str,
                              decision_source: str, session_id: str = None, tool_name: str = None,
                              confidence: float = None):
    """A model-routing decision (CW-6's decide_turn_model / CW-7's decide_tool_model), broadcast
    on the same "model" event_type over the existing Glass-box channel that scl-core-v2 already
    consumes via /trace-stream -- not a new event bus, the one broadcast_trace already serves.
    `decision_kind` is "turn" or "tool"; `decision_source` matches chat-wonder-v2-api#150's
    model_usage_log fields ("default" | "race_corrected" | "tool_override"). Call this from
    model_router.py once CW-6/CW-7 land -- this is the broadcast half, built ahead of the
    decision logic itself per the CW-10 SCL note."""
    # `text` (developer feed) carries the raw fields; `summary` (also reaches ilovelawyer-api's
    # scoped stream, since "model" is in _USER_TRACE_TYPES) stays human-readable prose, matching
    # every other broadcast_trace call site -- never a JSON blob.
    text = f"decision_kind={decision_kind} persona={persona} tool={tool_name} from_model={from_model} to_model={to_model} decision_source={decision_source} confidence={confidence}"
    if decision_kind == "tool" and tool_name:
        scope = f"the {tool_name} tool call"
    else:
        scope = "this turn"
    if from_model == to_model:
        summary = f"Kept {to_model} for {scope} ({decision_source})."
    else:
        summary = f"Switched {scope} from {from_model} to {to_model} ({decision_source})."
    broadcast_trace("model", text, session_id, summary=summary)


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


# Event types a customer may see, and the only fields of them that leave the server. `text` is
# developer-grade (session ids, raw tool output); `summary` is the plain-language explanation
# that broadcast_trace's callers already write for exactly this purpose. Metric events and
# anything without a summary are dropped.
_USER_TRACE_TYPES = frozenset({"request", "cognition", "action", "retrieval", "control", "memory", "model"})


def _user_trace_event(raw: str, session_id: str) -> Optional[str]:
    """Reduces one broadcast_trace payload to the customer-safe shape, or None to drop it."""
    try:
        event = json.loads(raw)
    except ValueError:
        return None
    if event.get("session_id") != session_id or event.get("type") not in _USER_TRACE_TYPES:
        return None
    summary = event.get("summary")
    if not summary:
        return None
    return json.dumps({"type": event["type"], "summary": summary, "turn_id": event.get("turn_id"), "ts": event.get("ts")})
