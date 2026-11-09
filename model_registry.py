# -*- coding: utf-8 -*-
"""Minimal multi-vendor dispatch for the Chat Completions loop.

Scope, deliberately: just what getting DeepSeek to actually work needs today
— which client a model id should use, and the one vendor-specific request
shape DeepSeek needs that GPT doesn't. This grows into the fuller per-tool/
per-persona registry later; not that yet.
"""

from __future__ import annotations

import os

from llm_usage import make_openai_client

_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
_deepseek_client = None


def is_deepseek_model(model: str) -> bool:
    return bool(model) and model.strip().lower().startswith("deepseek-")


def get_client(model: str, fallback_client):
    """fallback_client is state.openai_client, reused as-is for every OpenAI
    model id (same credentials/usage-tracking as today, zero behavior change).
    Only a deepseek-* model id gets routed to a separate client, built once
    and cached, so repeated calls don't re-authenticate."""
    if not is_deepseek_model(model):
        return fallback_client
    global _deepseek_client
    if _deepseek_client is None:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError(
                "model_registry: DEEPSEEK_API_KEY is not set; cannot call a deepseek-* model."
            )
        _deepseek_client = make_openai_client(api_key=api_key, base_url=_DEEPSEEK_BASE_URL)
    return _deepseek_client


def shape_chat_args(model: str, args: dict) -> dict:
    """Adjusts a Chat Completions args dict for the vendor `model` belongs to.

    DeepSeek's v4-pro/flash models default to thinking mode (enabled, effort
    "high") — per DeepSeek's own docs, once a request carries `tools`, every
    prior turn's `reasoning_content` must be echoed back on the next request
    or multi-turn tool calling breaks. This loop's message-building has no
    such round-trip, so force non-thinking mode instead: one-shot
    content/tool_calls per turn, the same shape GPT already runs in, and the
    documented, supported way to get that from DeepSeek
    (extra_body={"thinking": {"type": "disabled"}}).

    `reasoning_effort` is dropped for DeepSeek models in that case: it only
    controls thinking *effort*, not whether thinking happens, and is
    meaningless once thinking is off.
    """
    if not is_deepseek_model(model):
        return args
    args.pop("reasoning_effort", None)
    extra_body = dict(args.get("extra_body") or {})
    extra_body["thinking"] = {"type": "disabled"}
    args["extra_body"] = extra_body
    return args
