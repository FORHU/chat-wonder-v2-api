# -*- coding: utf-8 -*-
"""Model-string-driven provider resolution: OpenAI <-> DeepSeek.

Provider, client credentials, and request-arg quirks are all derived from the
model name string alone (e.g. "deepseek-v4-pro" implies DeepSeek). No separate
provider flag exists anywhere in this module or its callers.
"""

from __future__ import annotations

import os
import re
from typing import Any, Callable, Dict, Optional

from openai import OpenAI

_DEEPSEEK_BASE_URL_DEFAULT = "https://api.deepseek.com/v1"

_REASONING_CLASS_PREFIXES = (
    "gpt-5",
    "deepseek-reasoner",
    "deepseek-v4-pro",
)

_THINK_TAG_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def provider_for_model(model: Optional[str]) -> str:
    """Return "deepseek" if the model name is DeepSeek-branded, else "openai" (safe default)."""
    if model and model.strip().lower().startswith("deepseek-"):
        return "deepseek"
    return "openai"


def client_kwargs_for_model(model: Optional[str]) -> Dict[str, str]:
    """Resolve {"api_key": ..., "base_url": ...} for the given model's provider.

    base_url is omitted for OpenAI, matching init_openai_client's existing
    "omit if falsy" behavior.
    """
    if provider_for_model(model) == "deepseek":
        kwargs = {"api_key": os.getenv("DEEPSEEK_API_KEY", "")}
        base_url = os.getenv("DEEPSEEK_BASE_URL", _DEEPSEEK_BASE_URL_DEFAULT)
        if base_url:
            kwargs["base_url"] = base_url
        return kwargs
    return {"api_key": os.getenv("OPENAI_API_KEY", "")}


def is_reasoning_class(model: Optional[str]) -> bool:
    """True for models that reject the `temperature` param (gpt-5*, deepseek-reasoner, deepseek-v4-pro)."""
    if not model:
        return False
    lowered = model.strip().lower()
    return any(lowered.startswith(prefix) for prefix in _REASONING_CLASS_PREFIXES)


def normalize_request_kwargs(
    model: Optional[str], kwargs: Dict[str, Any], *, has_tools: bool = False
) -> Dict[str, Any]:
    """Return a copy of kwargs with provider-specific incompatibilities stripped.

    - drops `n` for DeepSeek (rejected)
    - drops `stop` if it is None
    - drops `temperature` when the model is reasoning-class
    - drops `reasoning_effort` when the provider isn't OpenAI (unsupported)
    - forces `parallel_tool_calls=False` when has_tools is True
    """
    out = dict(kwargs)
    provider = provider_for_model(model)

    if provider == "deepseek":
        out.pop("n", None)

    if "stop" in out and out["stop"] is None:
        out.pop("stop")

    if is_reasoning_class(model):
        out.pop("temperature", None)

    if provider != "openai":
        out.pop("reasoning_effort", None)

    if has_tools:
        out["parallel_tool_calls"] = False

    return out


def build_client(model: Optional[str]) -> OpenAI:
    """Construct an OpenAI-SDK client pointed at the right provider for `model`.

    Cheap to call per call site — no network call, matches the existing
    per-call client construction pattern already used across the codebase.
    """
    return OpenAI(**client_kwargs_for_model(model))


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from a complete string. No-op if none present."""
    if not text:
        return text
    return _THINK_TAG_RE.sub("", text)


def is_truncated_empty(response: Any) -> bool:
    """True if a completion hit its token limit before producing any content.

    Reasoning-class models (confirmed live for deepseek-v4-flash, not just the
    -pro/-reasoner models normalize_request_kwargs treats as reasoning-class) can
    spend their entire max_tokens budget on a separate `reasoning_content` field
    and return finish_reason="length" with an empty `content` — a silent failure
    that looks like a successful, merely-blank response unless checked for.
    """
    try:
        choice = response.choices[0]
    except (IndexError, AttributeError, TypeError):
        return False
    content = (choice.message.content or "").strip()
    return choice.finish_reason == "length" and not content


def make_think_tag_stream_filter() -> Callable[[str], str]:
    """Return a stateful filter for stripping <think>...</think> blocks from a token stream.

    A <think> block can span multiple stream chunks, so the filter buffers
    content while inside a block and only releases text once it knows whether
    that text falls outside any <think> tag.
    """
    state = {"inside": False, "buffer": ""}

    def _feed(chunk: str) -> str:
        if not chunk:
            return ""
        state["buffer"] += chunk
        out = []

        while True:
            buf = state["buffer"]
            if not state["inside"]:
                start = buf.find("<think>")
                if start == -1:
                    # No open tag yet; hold back a trailing partial "<think>" prefix.
                    keep = _partial_tag_suffix_len(buf, "<think>")
                    emit_len = len(buf) - keep
                    if emit_len > 0:
                        out.append(buf[:emit_len])
                        state["buffer"] = buf[emit_len:]
                    break
                out.append(buf[:start])
                state["buffer"] = buf[start + len("<think>"):]
                state["inside"] = True
            else:
                end = state["buffer"].find("</think>")
                if end == -1:
                    # Still inside the block; keep buffering, don't emit anything.
                    break
                state["buffer"] = state["buffer"][end + len("</think>"):]
                state["inside"] = False

        return "".join(out)

    return _feed


def _partial_tag_suffix_len(buf: str, tag: str) -> int:
    """Length of the longest suffix of buf that is a prefix of tag (possible split-tag)."""
    max_len = min(len(buf), len(tag) - 1)
    for length in range(max_len, 0, -1):
        if buf.endswith(tag[:length]):
            return length
    return 0
