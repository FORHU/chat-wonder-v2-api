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

# ALLOWLIST, not a denylist: an unknown model id gets no logprobs rather than risking a 400 from
# a provider that rejects the parameter. gpt-5.6-luna/terra (Responses API, no logprobs support)
# and deepseek-reasoner (hard error on logprobs, per DeepSeek's own docs) must never be added
# here — see confidence.py's module docstring for why this gap is the premise of that module,
# not a bug to work around.
_LOGPROB_CAPABLE_PREFIXES = ("gpt-4o", "gpt-4.1", "deepseek-chat")


def is_deepseek_model(model: str) -> bool:
    return bool(model) and model.strip().lower().startswith("deepseek-")


def supports_logprobs(model: str) -> bool:
    """Whether `model` can be asked for `logprobs`/`top_logprobs` on the Chat Completions API
    without erroring. Used by the_server.py's perform_chat to decide whether to request
    telemetry, and by shape_chat_args below as a defensive strip for any model that slips
    through regardless."""
    m = (model or "").strip().lower()
    return any(m.startswith(p) for p in _LOGPROB_CAPABLE_PREFIXES)


def logprob_capable_prefixes() -> tuple:
    """Public read of the allowlist, for /version to advertise which models can show real
    entropy — callers outside this module should use this rather than reaching into
    _LOGPROB_CAPABLE_PREFIXES directly."""
    return _LOGPROB_CAPABLE_PREFIXES


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

    Also strips `logprobs`/`top_logprobs` for any model outside supports_logprobs' allowlist —
    a defensive belt-and-suspenders check: perform_chat only sets them when supports_logprobs()
    already agreed, but this guarantees a stray flag can never 400 a real turn regardless of how
    it got into `args`.
    """
    if not supports_logprobs(model):
        args.pop("logprobs", None)
        args.pop("top_logprobs", None)
    if not is_deepseek_model(model):
        return args
    args.pop("reasoning_effort", None)
    extra_body = dict(args.get("extra_body") or {})
    extra_body["thinking"] = {"type": "disabled"}
    args["extra_body"] = extra_body
    return args
