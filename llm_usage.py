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

    def add(self, usage: Any, model: Optional[str]) -> None:
        """Add one call's usage. Accepts Chat Completions and Responses shapes."""
        try:
            prompt = _int(_get(usage, "prompt_tokens", _get(usage, "input_tokens")))
            completion = _int(_get(usage, "completion_tokens", _get(usage, "output_tokens")))
            details = _get(usage, "prompt_tokens_details") or _get(usage, "input_tokens_details")
            cached = _int(_get(details, "cached_tokens"))
            total = _int(_get(usage, "total_tokens")) or prompt + completion
            key = model or "unknown"
            with self._lock:
                row = self._by_model.setdefault(
                    key, {"prompt_tokens": 0, "cached_input_tokens": 0,
                          "completion_tokens": 0, "total_tokens": 0, "calls": 0})
                row["prompt_tokens"] += prompt
                row["cached_input_tokens"] += cached
                row["completion_tokens"] += completion
                row["total_tokens"] += total
                row["calls"] += 1
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
            "by_model": [
                {"model": r["model"], "prompt_tokens": r["prompt_tokens"],
                 "cached_input_tokens": r["cached_input_tokens"],
                 "completion_tokens": r["completion_tokens"],
                 "total_tokens": r["total_tokens"], "calls": r["calls"]}
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
    """Yield the stream unchanged, except the usage-only final chunk.

    With include_usage the last chunk has empty `choices`; callers do
    chunk.choices[0], so it is recorded and swallowed here.
    """
    model = requested_model
    for chunk in stream:
        model = _get(chunk, "model") or model
        usage = _get(chunk, "usage")
        if usage is not None and not _get(chunk, "choices"):
            turn.add(usage, model)
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
