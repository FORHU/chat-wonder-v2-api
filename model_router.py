# -*- coding: utf-8 -*-
"""Turn-start and per-tool model decisions (CW-6 chat-wonder-v2-api#147, CW-7 #148), built on
decisions_research.py's proven Decisions API call shape rather than a second implementation of
it.

Scope, deliberately: this module answers "which model should serve this call" as a pure
decision. It does NOT do the actual racing (firing the default `call_llm` immediately via
asyncio.gather alongside this decision, discarding and restarting via legal_verify.py's
__DRAFT_DISCARD__ only on disagreement) -- that orchestration belongs in the_server.py's loop,
and CW-6's own checklist requires a p50/p95/p99 latency benchmark with the race in place before
MODEL_DECISION_TIMEOUT_MS's default is picked, let alone before this is wired into a real turn.
Building that integration ahead of the benchmark would ship exactly the thing the plan calls out
as unproven ("racing beats gating" -- PR #139's own numbers went 102s -> 107s despite fewer
model calls).

Fails open everywhere, same posture as decisions_research.py: any error, timeout, or
low-confidence answer means "keep the default model", never "block the turn" or "guess".

Precedence, highest first:
  1. A per-tool pin in this persona's routing override (routing_overrides.get_override, CW-8) --
     only for decide_tool_model.
  2. The persona's routing override set to MANUAL (routing_overrides.get_override, CW-8) -- no
     Decisions API call at all.
  3. The Decisions API, only when routingMode is AUTO (and, for a tool call, only when the tool
     is in MODEL_SENSITIVE_TOOLS -- empty by default, so no tool pays for this unless opted in).
  4. The default model.
"""
from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional

import decisions_research as dr
import model_registry as mr
import routing_overrides as ro

_logger = logging.getLogger(__name__)


def _broadcast(**kwargs):
    # Deferred import, same reason legal_responses_chain.py imports from the_server inside
    # function bodies: the_server.py is the entry point and will eventually import this module,
    # so a top-level import here would be circular.
    from the_server import broadcast_model_decision

    broadcast_model_decision(**kwargs)

# Provisional -- same posture as decisions_research.py's SECTION_HELPS_MIN/ACT_CENTRAL_MIN
# ("provisional, not trusted as-is"). To be set from real comparison data, not guessed.
MODEL_DECISION_CONFIDENCE_MIN = 0.5


def _model_decision_timeout_s() -> float:
    # Deliberately its own env var, not decisions_research.DECISIONS_TIMEOUT_S: that default
    # (20s) is sized for a sequential prefetch, not a race against an in-flight stream, where
    # it would make the race meaningless (the default stream usually finishes first anyway).
    # CW-6's own acceptance criteria require benchmarking this before it's locked in --
    # provisional until that benchmark exists.
    try:
        return float(os.getenv("MODEL_DECISION_TIMEOUT_MS", "3000")) / 1000.0
    except ValueError:
        return 3.0


def _sensitive_tools() -> frozenset:
    raw = os.getenv("MODEL_SENSITIVE_TOOLS", "")
    return frozenset(t.strip() for t in raw.split(",") if t.strip())


def _compatible_candidates(candidates: List[str], require_responses_api_ok: bool) -> List[str]:
    """Drops a candidate the current call shape cannot actually use. `require_responses_api_ok`
    is True when the caller is on the Chat Completions path with tools+a real reasoning_effort
    in play -- a model that requires /v1/responses for that combination can't be switched to
    mid-call without also switching API shape, which is out of scope here."""
    if not require_responses_api_ok:
        return candidates
    return [m for m in candidates if not mr.model_spec(m).requires_responses_api_with_reasoning_tools]


def _best_candidate(client, query: str, options: List[str]) -> "tuple[Optional[str], float]":
    """One Decisions API request, a predicate per candidate ("would this model be clearly
    better than the default?"), independent rather than mutually exclusive -- there are
    usually one or two real alternatives, not a large mutually-exclusive category set like
    decisions_research.py's Act ranking. Raises on failure; callers fail open."""
    questions = [
        {
            "name": f"model{i}",
            "type": "predicate",
            "instructions": f"Given the user's question, would {m!r} be a clearly better choice of model to answer it than the current default?",
        }
        for i, m in enumerate(options)
    ]
    answers = dr.decide(client, query, questions)
    best_model, best_p = None, 0.0
    for i, m in enumerate(options):
        p = (answers.get(f"model{i}") or {}).get("probability", 0.0)
        if p > best_p:
            best_model, best_p = m, p
    return best_model, best_p


def _decision_result(model: str, decision_source: str, confidence: Optional[float]) -> Dict:
    return {"model": model, "decision_source": decision_source, "confidence": confidence}


def decide_turn_model(
    client,
    persona: str,
    default_model: str,
    candidate_models: Optional[List[str]] = None,
    query: str = "",
    session_id: Optional[str] = None,
    require_responses_api_ok: bool = False,
) -> Dict:
    """CW-6. `candidate_models` are alternates the Decisions API may switch to; the persona's
    MANUAL override (if any) always wins over them. Broadcasts the outcome either way
    (the_server.broadcast_model_decision) so it's visible on the live trace even when nothing
    changed."""
    override = ro.get_override(persona)
    if override.get("routingMode") == "MANUAL" and override.get("manualPinnedModel"):
        chosen = override["manualPinnedModel"]
        _broadcast(persona=persona, decision_kind="turn", from_model=default_model, to_model=chosen, decision_source="manual", session_id=session_id)
        return _decision_result(chosen, "manual", None)

    options = _compatible_candidates([m for m in (candidate_models or []) if m != default_model], require_responses_api_ok)
    if not options:
        return _decision_result(default_model, "default", None)

    try:
        best_model, best_p = _best_candidate(client, query, options)
    except Exception as e:  # noqa: BLE001 -- a Decisions API failure must never block or alter the turn
        _logger.warning("decide_turn_model failed for persona=%s, keeping default: %s", persona, e)
        return _decision_result(default_model, "default", None)

    if best_model is None or best_p < MODEL_DECISION_CONFIDENCE_MIN:
        return _decision_result(default_model, "default", best_p if best_model else None)

    _broadcast(persona=persona, decision_kind="turn", from_model=default_model, to_model=best_model, decision_source="race_corrected", session_id=session_id, confidence=best_p)
    return _decision_result(best_model, "race_corrected", best_p)


def decide_tool_model(
    client,
    persona: str,
    tool_name: str,
    default_model: str,
    candidate_models: Optional[List[str]] = None,
    query: str = "",
    session_id: Optional[str] = None,
    require_responses_api_ok: bool = False,
) -> Dict:
    """CW-7. A tool-specific pin in the persona's override always wins; otherwise identical
    precedence to decide_turn_model, gated by MODEL_SENSITIVE_TOOLS (empty by default -- an
    unflagged tool makes zero Decisions API calls, zero latency/cost, by construction)."""
    override = ro.get_override(persona)
    tool_overrides = override.get("toolOverrides") or {}
    if tool_name in tool_overrides and tool_overrides[tool_name]:
        chosen = tool_overrides[tool_name]
        _broadcast(persona=persona, decision_kind="tool", from_model=default_model, to_model=chosen, decision_source="tool_override", session_id=session_id, tool_name=tool_name)
        return _decision_result(chosen, "tool_override", None)

    if override.get("routingMode") == "MANUAL" and override.get("manualPinnedModel"):
        chosen = override["manualPinnedModel"]
        _broadcast(persona=persona, decision_kind="tool", from_model=default_model, to_model=chosen, decision_source="manual", session_id=session_id, tool_name=tool_name)
        return _decision_result(chosen, "manual", None)

    if tool_name not in _sensitive_tools():
        return _decision_result(default_model, "default", None)

    options = _compatible_candidates([m for m in (candidate_models or []) if m != default_model], require_responses_api_ok)
    if not options:
        return _decision_result(default_model, "default", None)

    try:
        best_model, best_p = _best_candidate(client, query, options)
    except Exception as e:  # noqa: BLE001
        _logger.warning("decide_tool_model failed for persona=%s tool=%s, keeping default: %s", persona, tool_name, e)
        return _decision_result(default_model, "default", None)

    if best_model is None or best_p < MODEL_DECISION_CONFIDENCE_MIN:
        return _decision_result(default_model, "default", best_p if best_model else None)

    _broadcast(persona=persona, decision_kind="tool", from_model=default_model, to_model=best_model, decision_source="race_corrected", session_id=session_id, tool_name=tool_name, confidence=best_p)
    return _decision_result(best_model, "race_corrected", best_p)
