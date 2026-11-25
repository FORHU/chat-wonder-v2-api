# -*- coding: utf-8 -*-
"""Per-turn LLM token usage (chat-wonder-v2-api#104).

ilovelawyer-api needs the real token counts for every turn so it can store them
in its LlmUsage ledger (epic #399). One TurnUsage lives in a ContextVar for the
length of a turn; every OpenAI call made on that turn adds to it, and the
request handlers send it back as the `usage` block.

Why a ContextVar and not a field on ChatState: about 20 call sites build their
own OpenAI client and never touch state.openai_client, and calls run on worker
threads (asyncio.to_thread, _astream_llm). A ContextVar follows the turn into
asyncio.to_thread by itself; for a plain Thread, run it under
contextvars.copy_context().run. It also keeps usage out of the pickled
sessions.

Not counted: images.edit and audio.transcriptions. They are not token-billed
chat calls, so they are delegated untouched.

Recording never raises and never changes what the caller receives.
"""

from __future__ import annotations

import threading
from contextvars import ContextVar
from typing import Any, Optional


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _int(value: Any) -> int:
    try:
        return int(value) if value is not None else 0
    except (TypeError, ValueError):
        return 0


class TurnUsage:
    """Token totals for one turn, split by model."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_model: dict[str, dict[str, int]] = {}
        self.pass_count = 0
        self._default_routing_mode: Optional[str] = None
        self._default_decision_source: Optional[str] = None

    def set_routing_info(self, routing_mode: Optional[str], decision_source: Optional[str]) -> None:
        """Sticky per-turn default, set once right after a routing decision is made (the_server.py's
        _legal_model_override/_intake_model_override). `add()`'s own call sites (the ~20 OpenAI-call
        wrappers in this module) have no idea a routing decision happened, so they can't pass these
        themselves -- every add() for this turn picks these up automatically unless it passes its own
        (it never does today, but the override exists so a future call site still can)."""
        with self._lock:
            self._default_routing_mode = routing_mode
            self._default_decision_source = decision_source

    def add(
        self,
        usage: Any,
        model: Optional[str],
        *,
        routing_mode: Optional[str] = None,
        decision_source: Optional[str] = None,
    ) -> None:
        """Add one call's usage. Accepts Chat Completions and Responses shapes.

        `routing_mode`/`decision_source` are CW-9's addition (chat-wonder-v2-api#150): which
        routing_overrides.py mode was in effect and what model_router.py decided, for this
        model's calls this turn. Optional and None by default -- every existing call site (about
        20, per this module's own docstring) stays unchanged, since model_router.py isn't wired
        into any of them yet. When given more than once for the same model in one turn, the
        latest non-None value wins -- a later call that omits them (an untouched call site)
        never erases what an earlier, routing-aware call site already recorded. In practice
        every call to a given model within a turn comes from the same routing decision, so this
        never actually differs call to call."""
        try:
            prompt = _int(_get(usage, "prompt_tokens", _get(usage, "input_tokens")))
            completion = _int(_get(usage, "completion_tokens", _get(usage, "output_tokens")))
            details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
            cached = _int(_get(details, "cached_tokens"))
            total = _int(_get(usage, "total_tokens")) or prompt + completion
            key = model or "unknown"
            with self._lock:
                routing_mode = routing_mode if routing_mode is not None else self._default_routing_mode
                decision_source = decision_source if decision_source is not None else self._default_decision_source
                row = self._by_model.setdefault(
                    key, {"prompt_tokens": 0, "cached_input_tokens": 0,
                          "completion_tokens": 0, "total_tokens": 0, "calls": 0,
                          "routing_mode": None, "decision_source": None})
                row["prompt_tokens"] += prompt
                row["cached_input_tokens"] += cached
                row["completion_tokens"] += completion
                row["total_tokens"] += total
                row["calls"] += 1
                if routing_mode is not None:
                    row["routing_mode"] = routing_mode
                if decision_source is not None:
                    row["decision_source"] = decision_source
        except Exception:
            pass

    def add_pass(self) -> None:
        with self._lock:
            self.pass_count += 1

    def to_dict(self) -> dict:
        with self._lock:
            by_model = [{"model": m, **row} for m, row in self._by_model.items()]
            passes = self.pass_count
        by_model.sort(key=lambda r: r["prompt_tokens"], reverse=True)
        totals = {k: sum(r[k] for r in by_model) for k in
                  ("prompt_tokens", "cached_input_tokens", "completion_tokens", "total_tokens", "calls")}
        return {
            "prompt_tokens": totals["prompt_tokens"],
            "cached_input_tokens": totals["cached_input_tokens"],
            "completion_tokens": totals["completion_tokens"],
            "total_tokens": totals["total_tokens"],
            "call_count": totals["calls"],
            "pass_count": passes,
            "model": by_model[0]["model"] if by_model else None,
            # Top-level, alongside `model` (same "dominant model this turn" row) rather than only
            # inside by_model -- ilovelawyer-api's ingestion (chat-wonder-usage.ts) records one
            # LlmUsage row per call from the flat top-level fields, it does not read by_model.
            "routing_mode": by_model[0].get("routing_mode") if by_model else None,
            "decision_source": by_model[0].get("decision_source") if by_model else None,
            "by_model": [
                {"model": r["model"], "prompt_tokens": r["prompt_tokens"],
                 "cached_input_tokens": r["cached_input_tokens"],
                 "completion_tokens": r["completion_tokens"],
                 "total_tokens": r["total_tokens"], "calls": r["calls"],
                 "routing_mode": r.get("routing_mode"), "decision_source": r.get("decision_source")}
                for r in by_model
            ],
        }


_current: ContextVar[Optional[TurnUsage]] = ContextVar("turn_usage", default=None)


def start_turn() -> TurnUsage:
    """Begin a turn with a fresh, zeroed TurnUsage."""
    turn = TurnUsage()
    _current.set(turn)
    return turn


def current() -> TurnUsage:
    """The current turn's usage. Outside a turn, a throwaway object so callers need no None check."""
    turn = _current.get()
    return turn if turn is not None else TurnUsage()


def record_response_event(turn: TurnUsage, event: Any) -> None:
    """Responses API streams: usage rides on the response.completed event."""
    try:
        if _get(event, "type") == "response.completed":
            resp = _get(event, "response")
            turn.add(_get(resp, "usage"), _get(resp, "model"))
    except Exception:
        pass


def _tracked_stream(stream, turn: TurnUsage, requested_model: Optional[str]):
    """Yield the stream unchanged, except a usage-only chunk with no real content.

    OpenAI's include_usage sends usage on its own trailing chunk with empty `choices` --
    callers do chunk.choices[0], so that chunk is recorded and swallowed (never yielded).
    DeepSeek's OpenAI-compatible API instead attaches usage to the SAME chunk as the final
    content delta, which has real, non-empty `choices` (confirmed live: finish_reason="stop"
    alongside a populated usage object) -- swallowing that chunk would drop real content the
    caller needs, so it's recorded AND yielded. Either way, usage is captured whenever present;
    only an empty-choices chunk is ever withheld from the caller.
    """
    model = requested_model
    for chunk in stream:
        model = _get(chunk, "model") or model
        usage = _get(chunk, "usage")
        if usage is not None:
            turn.add(usage, model)
            if not _get(chunk, "choices"):
                continue
        yield chunk


class _CompletionsProxy:
    def __init__(self, inner):
        self._inner = inner

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def create(self, *args, **kwargs):
        turn = current()  # captured now: the stream may be consumed on another thread
        streaming = bool(kwargs.get("stream"))
        if streaming and "stream_options" not in kwargs:
            kwargs["stream_options"] = {"include_usage": True}
        result = self._inner.create(*args, **kwargs)
        try:
            if streaming:
                if (kwargs.get("stream_options") or {}).get("include_usage"):
                    return _tracked_stream(result, turn, kwargs.get("model"))
                return result
            turn.add(_get(result, "usage"), _get(result, "model") or kwargs.get("model"))
        except Exception:
            pass
        return result


class _ChatProxy:
    def __init__(self, inner):
        self._inner = inner
        self.completions = _CompletionsProxy(inner.completions)

    def __getattr__(self, name):
        return getattr(self._inner, name)


class UsageTrackingOpenAI:
    """Delegates to an OpenAI client, recording chat.completions.create usage."""

    def __init__(self, client):
        self._client = client
        self.chat = _ChatProxy(client.chat)

    def __getattr__(self, name):
        return getattr(self._client, name)


def make_openai_client(**kwargs) -> UsageTrackingOpenAI:
    from openai import OpenAI

    return UsageTrackingOpenAI(OpenAI(**kwargs))
