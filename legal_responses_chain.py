# -*- coding: utf-8 -*-
"""Legal-persona tool-calling loop on OpenAI's Responses API.

Exists because gpt-5.6-terra rejects function tools combined with any
reasoning_effort other than 'none' on /v1/chat/completions (confirmed live):
"Function tools with reasoning_effort are not supported for gpt-5.6-terra in
/v1/chat/completions. To use function tools, use /v1/responses or set
reasoning_effort to 'none'." This module is the /v1/responses path, used only
for the legal persona so every other persona keeps using Chat Completions
unchanged.

Confirmed live (Stage 0 spike) before writing this:
- No `reasoning` output items are returned by gpt-5.6-terra even at
  effort='high', and replaying only function_call/function_call_output pairs
  (no reasoning item) round-trips into a coherent next answer. No reasoning
  item persistence is implemented here because there is nothing to persist.
- temperature is rejected outright ("Unsupported parameter: 'temperature' is
  not supported with this model.") -- never pass it to responses.create for
  this model.
- The inline `REASON: ...` prompt convention (see prepare_chat_messages's
  [XAI Transparency Requirement] block) still works verbatim as plain
  message text under effort='high' -- reused unchanged by the caller.
"""

from __future__ import annotations

import asyncio
import json
from typing import List, Optional


def to_responses_tools(chat_completions_tools: Optional[list]) -> list:
    """Flatten {"type": "function", "function": {...}} -> Responses tool shape."""
    if not chat_completions_tools:
        return []
    flattened = []
    for t in chat_completions_tools:
        fn = t.get("function") if isinstance(t, dict) and "function" in t else t
        if not isinstance(fn, dict) or "name" not in fn:
            continue
        flattened.append(
            {
                "type": "function",
                "name": fn["name"],
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
            }
        )
    return flattened


def chat_messages_to_input_items(messages: List[dict]) -> List[dict]:
    """Convert prepare_chat_messages()'s flat role/content list into Responses
    input items. Confirmed live: role/content items (including 'system' mid-list,
    which this app uses for synthetic [Memory Fact]/[Constraints] narration)
    pass through Responses API unchanged -- this is intentionally a thin,
    named passthrough rather than dead identity code, so the seam exists if
    that ever stops being true for a future model.
    """
    return [dict(m) for m in messages]


def append_function_call_round_trip(
    input_items: List[dict],
    *,
    call_id: str,
    name: str,
    arguments: str,
    result,
) -> List[dict]:
    """Append the function_call + function_call_output item pair, replacing
    the Chat-Completions-loop's synthetic '[Memory Fact]' system-message trick.
    """
    try:
        output = result if isinstance(result, str) else json.dumps(result, ensure_ascii=False)
    except Exception:
        output = str(result)
    input_items.append(
        {
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        }
    )
    input_items.append(
        {
            "type": "function_call_output",
            "call_id": call_id,
            "output": output,
        }
    )
    return input_items


def _first_pending_call(pending_calls: dict):
    """pending_calls is keyed by output_index; we only ever act on one call per
    turn (matching the existing single-call-per-turn assumption used throughout
    the_server.py's dedup/trace/HITL logic). Take the lowest index; warn if more
    than one arrived so a future model that does parallelize here is visible in
    logs rather than silently dropping calls -- same posture as the
    parallel_tool_calls=False comment on the Chat Completions loop.
    """
    if not pending_calls:
        return None
    if len(pending_calls) > 1:
        import logging

        logging.warning(
            "[legal-responses] model emitted %d parallel tool calls in one turn; "
            "only the first is executed (matches existing single-call assumption)",
            len(pending_calls),
        )
    first_index = min(pending_calls.keys())
    return pending_calls[first_index]


def apply_responses_event(event, pending_calls: dict) -> Optional[str]:
    """Process one Responses API stream event, mutating pending_calls in place
    (keyed by output_index -- item-indexed, unlike Chat Completions' flat
    delta.tool_calls[0]). Returns a text delta string if this event carried
    visible text, else None. Used identically by the sync and async loops.
    """
    et = event.type
    if et == "response.output_item.added" and getattr(event.item, "type", None) == "function_call":
        pending_calls[event.output_index] = {
            "call_id": event.item.call_id,
            "name": event.item.name,
            "arguments": "",
        }
    elif et == "response.function_call_arguments.delta":
        if event.output_index in pending_calls:
            pending_calls[event.output_index]["arguments"] += event.delta
    elif et == "response.output_item.done" and getattr(event.item, "type", None) == "function_call":
        # .done's item.arguments is the authoritative complete string;
        # reconcile in case any delta was dropped (defensive, .added already
        # carries call_id/name per the Stage 0 spike).
        pending_calls[event.output_index] = {
            "call_id": event.item.call_id,
            "name": event.item.name,
            "arguments": event.item.arguments,
        }
    elif et == "response.output_text.delta":
        return event.delta
    return None


def _strip_reason_line(text: str):
    """Same convention as the Chat Completions loop: a leading 'REASON: ...'
    line is system-logging only and never shown to the user. Operates on a
    fully-accumulated string (sync loop calls this once per iteration; the
    streaming loop buffers the first line itself before deciding what to
    yield -- see streaming_run_function_chain_responses).
    """
    if not text:
        return text, None
    lines = text.split("\n")
    if lines[0].strip().startswith("REASON:"):
        return "\n".join(lines[1:]).strip(), lines[0].strip()[7:].strip()
    return text, None


def run_function_chain_responses(
    state,
    messages: list,
    max_chains: int = 12,
    session_id: str = None,
    tools: list = None,
    query: str = "",
    model: str = None,
    reasoning_effort: str = None,
    temperature: float = None,
):
    """Legal-persona equivalent of the_server.run_function_chain(), on /v1/responses
    instead of /v1/chat/completions, so real reasoning_effort can be combined
    with tool calling (gpt-5.6-terra rejects that combination on Chat Completions).

    temperature is accepted for call-site signature symmetry with the Chat
    Completions loop but never sent -- confirmed live that this model rejects
    it outright ("Unsupported parameter: 'temperature' is not supported with
    this model.").
    """
    from the_server import (
        _context,
        _describe_tool_args,
        _summarize_tool_result,
        broadcast_trace,
        execute_function_call,
    )

    available_manifest = to_responses_tools(tools if tools is not None else _context.fun_manifest)
    input_items = chat_messages_to_input_items(messages)
    funcall_chains = []
    full_response = ""
    last_tool = None

    def perform_responses_call(items):
        args = {
            "model": model or _context.model,
            "input": items,
            "stream": True,
        }
        if reasoning_effort and reasoning_effort != "none":
            args["reasoning"] = {"effort": reasoning_effort}
        if available_manifest:
            args["tools"] = available_manifest
            args["tool_choice"] = "auto"
            args["parallel_tool_calls"] = False
        return state.openai_client.responses.create(**args)

    for _ in range(max_chains):
        if last_tool:
            _cycle_summary = f"The AI received results from '{last_tool}' and is deciding whether it has enough information to answer or needs to take another step."
        else:
            _cycle_summary = "The AI is working through the question, deciding whether it needs to use a tool or can answer directly."
        broadcast_trace(
            "cognition",
            f"Cycle {_ + 1} — reasoning over {len(input_items)} items (model: {model}, reasoning_effort: {reasoning_effort})",
            session_id,
            summary=_cycle_summary,
        )

        pending_calls: dict = {}
        last_response = ""
        for event in perform_responses_call(input_items):
            delta_text = apply_responses_event(event, pending_calls)
            if delta_text:
                last_response += delta_text

        function_call = _first_pending_call(pending_calls)
        last_response, _xai_reason = _strip_reason_line(last_response.strip())

        if not function_call and last_response:
            preview = last_response[:200].replace("\n", " ")
            broadcast_trace(
                "cognition",
                f"LLM produced final text ({len(last_response)} chars): \"{preview}{'…' if len(last_response) > 200 else ''}\"",
                session_id,
                summary="The AI has finished reasoning and is ready to deliver its response.",
            )

        if _xai_reason and function_call:
            broadcast_trace(
                "cognition",
                f"Reasoning: {_xai_reason}",
                session_id,
                summary=f"In the AI's own words, it explained its decision: \"{_xai_reason}\"",
            )

        if function_call:
            _tool_desc = next(
                (t["description"] for t in available_manifest if t["name"] == function_call["name"]), ""
            )
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

        if not function_call:
            break

        # HITL gate
        if not _context.auto_approval:
            return {"__hitl__": True, "function_call": function_call, "messages": input_items, "tools": tools}

        try:
            cur_args = json.loads(function_call["arguments"])
        except Exception:
            cur_args = function_call["arguments"]

        is_dup = any(
            fc["name"] == function_call["name"] and fc["args"] == cur_args for fc in funcall_chains
        )
        if is_dup:
            broadcast_trace(
                "control",
                f"BLOCKED duplicate call: `{function_call['name']}` — injecting memory reminder",
                session_id,
                summary=f"Safety check: The AI proposed to use '{function_call['name']}' again with the same inputs. This was blocked to prevent a redundant loop.",
            )
            input_items.append({
                "role": "system",
                "content": f"Function `{function_call['name']}` already called with same args. Do not repeat.",
            })
            continue

        broadcast_trace(
            "control",
            f"APPROVED: `{function_call['name']}` — no prior identical call found",
            session_id,
            summary=f"Safety check passed. The AI's proposed action is new — it has not taken this exact step before. Proceeding to execute '{function_call['name']}'.",
        )
        broadcast_trace(
            "action",
            f"Executing `{function_call['name']}`...",
            session_id,
            summary=f"The AI is now running '{function_call['name']}' to retrieve the information it needs.",
        )

        funcall_chains.append({"name": function_call["name"], "args": cur_args})
        result = execute_function_call(function_call, session_id=session_id)
        if result is None:
            continue
        last_tool = function_call["name"]
        state.turn_tool_calls = getattr(state, "turn_tool_calls", 0) + 1

        try:
            _rp = json.dumps(result, ensure_ascii=False)
        except Exception:
            _rp = str(result)
        _ctx = _summarize_tool_result(function_call["name"], result)
        broadcast_trace(
            "action",
            f"Result from `{function_call['name']}`:\n{_rp[:300]}",
            session_id,
            summary=f"'{function_call['name']}' completed. {_ctx}",
        )
        broadcast_trace(
            "memory",
            f"Fact stored: `{function_call['name']}` result is now confirmed knowledge.\nValue: {_rp[:150]}",
            session_id,
            summary=f"The AI stored the result from '{function_call['name']}'. This confirmed knowledge will be used when composing the final response.",
        )

        append_function_call_round_trip(
            input_items,
            call_id=function_call["call_id"],
            name=function_call["name"],
            arguments=function_call["arguments"],
            result=result,
        )
        input_items.append({
            "role": "system",
            "content": "[Constraints]\nIf a complete response has been produced, TERMINATE. Only execute new actions if their conditions are fully satisfied.",
        })

    if not full_response and funcall_chains:
        # Exhausted max_chains while the model still wanted to call tools --
        # force one final tool-free completion instead of returning nothing
        # (same posture as the Chat Completions loop's identical fallback).
        try:
            args = {
                "model": model or _context.model,
                "input": input_items + [{
                    "role": "system",
                    "content": (
                        "[Constraints]\nYou have reached the maximum number of tool calls "
                        "for this turn. Do NOT call any more tools. Answer now using only "
                        "the information already gathered above, and say clearly if some "
                        "aspect could not be fully verified."
                    ),
                }],
            }
            if reasoning_effort and reasoning_effort != "none":
                args["reasoning"] = {"effort": reasoning_effort}
            forced = state.openai_client.responses.create(**args)
            full_response = (forced.output_text or "").strip()
        except Exception as e:
            import logging

            logging.warning("Forced final-answer completion (responses API) failed: %s", e)

    return full_response


async def streaming_run_function_chain_responses(
    state,
    messages: list,
    max_chains: int = 12,
    session_id: str = None,
    tools: list = None,
    query: str = "",
    model: str = None,
    reasoning_effort: str = None,
    temperature: float = None,
):
    """Streaming (async generator) counterpart of run_function_chain_responses,
    structurally mirroring the_server.streaming_run_function_chain -- same
    REASON:-line buffering/withholding behavior, same trace points, reusing
    the_server._astream_llm (generic sync-stream-to-async-generator bridge,
    not tied to Chat Completions' event shape) to bridge the Responses stream.
    """
    from the_server import (
        _astream_llm,
        _context,
        _describe_tool_args,
        _summarize_tool_result,
        broadcast_trace,
        execute_function_call,
    )

    available_manifest = to_responses_tools(tools if tools is not None else _context.fun_manifest)
    input_items = chat_messages_to_input_items(messages)
    funcall_chains = []
    full_response = ""
    last_tool = None

    def perform_responses_call(items):
        args = {
            "model": model or _context.model,
            "input": items,
            "stream": True,
        }
        if reasoning_effort and reasoning_effort != "none":
            args["reasoning"] = {"effort": reasoning_effort}
        if available_manifest:
            args["tools"] = available_manifest
            args["tool_choice"] = "auto"
            args["parallel_tool_calls"] = False
        return state.openai_client.responses.create(**args)

    for iteration in range(max_chains):
        if last_tool:
            _cycle_summary = f"The AI received results from '{last_tool}' and is deciding whether it has enough information to answer or needs to take another step."
        else:
            _cycle_summary = "The AI is working through the question, deciding whether it needs to use a tool or can answer directly."
        broadcast_trace(
            "cognition",
            f"Cycle {iteration + 1} — reasoning over {len(input_items)} items (model: {model}, reasoning_effort: {reasoning_effort})",
            session_id,
            summary=_cycle_summary,
        )
        await asyncio.sleep(0)

        pending_calls: dict = {}
        _xai_buffer = ""
        _xai_first_line_done = False
        _xai_reason = None
        last_response = ""

        async for event in _astream_llm(perform_responses_call, input_items):
            delta_text = apply_responses_event(event, pending_calls)
            if not delta_text:
                continue
            part = delta_text.replace("~", "-")
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

        function_call = _first_pending_call(pending_calls)

        if function_call:
            import logging

            logging.info("chain[%d] LLM→tool=%s session=%s", iteration, function_call["name"], session_id)
        else:
            import logging

            logging.info("chain[%d] LLM→text chars=%d session=%s", iteration, len(last_response), session_id)

        if not function_call and last_response:
            preview = last_response[:200].replace("\n", " ")
            broadcast_trace(
                "cognition",
                f"LLM produced final text ({len(last_response)} chars): \"{preview}{'…' if len(last_response) > 200 else ''}\"",
                session_id,
                summary="The AI has finished reasoning and is ready to deliver its response.",
            )
            await asyncio.sleep(0)

        if _xai_reason and function_call:
            broadcast_trace(
                "cognition",
                f"Reasoning: {_xai_reason}",
                session_id,
                summary=f"In the AI's own words, it explained its decision: \"{_xai_reason}\"",
            )
            await asyncio.sleep(0)

        if function_call:
            _tool_desc = next(
                (t["description"] for t in available_manifest if t["name"] == function_call["name"]), ""
            )
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

        if not function_call:
            break

        # HITL gate: emit pending_approval event and stop streaming
        if not _context.auto_approval:
            yield f"__HITL__{json.dumps({'function_call': function_call, 'messages': input_items, 'tools': [t['name'] for t in (available_manifest or [])]})}"
            return

        try:
            cur_args = json.loads(function_call["arguments"])
        except Exception:
            cur_args = function_call["arguments"]

        is_dup = any(
            fc["name"] == function_call["name"] and fc["args"] == cur_args for fc in funcall_chains
        )
        if is_dup:
            broadcast_trace(
                "control",
                f"BLOCKED duplicate call: `{function_call['name']}` — injecting memory reminder",
                session_id,
                summary=f"Safety check: The AI proposed to use '{function_call['name']}' again with the same inputs. This was blocked to prevent a redundant loop.",
            )
            await asyncio.sleep(0)
            input_items.append({
                "role": "system",
                "content": f"Function `{function_call['name']}` already called. Do not repeat.",
            })
            continue

        broadcast_trace(
            "control",
            f"APPROVED: `{function_call['name']}` — no prior identical call found",
            session_id,
            summary=f"Safety check passed. The AI's proposed action is new — it has not taken this exact step before. Proceeding to execute '{function_call['name']}'.",
        )
        await asyncio.sleep(0)
        broadcast_trace(
            "action",
            f"Executing `{function_call['name']}`...",
            session_id,
            summary=f"The AI is now running '{function_call['name']}' to retrieve the information it needs.",
        )
        await asyncio.sleep(0)

        funcall_chains.append({"name": function_call["name"], "args": cur_args})
        result = await asyncio.to_thread(execute_function_call, function_call, session_id=session_id)
        if result is None:
            continue
        last_tool = function_call["name"]
        state.turn_tool_calls = getattr(state, "turn_tool_calls", 0) + 1

        try:
            _rp = json.dumps(result, ensure_ascii=False)
        except Exception:
            _rp = str(result)
        _ctx = _summarize_tool_result(function_call["name"], result)
        broadcast_trace(
            "action",
            f"Result from `{function_call['name']}`:\n{_rp[:300]}",
            session_id,
            summary=f"'{function_call['name']}' completed. {_ctx}",
        )
        await asyncio.sleep(0)
        broadcast_trace(
            "memory",
            f"Fact stored: `{function_call['name']}` result is now confirmed knowledge.\nValue: {_rp[:150]}",
            session_id,
            summary=f"The AI stored the result from '{function_call['name']}'. This confirmed knowledge will be used when composing the final response.",
        )
        await asyncio.sleep(0)

        append_function_call_round_trip(
            input_items,
            call_id=function_call["call_id"],
            name=function_call["name"],
            arguments=function_call["arguments"],
            result=result,
        )
        input_items.append({
            "role": "system",
            "content": "[Constraints]\nIf a complete response has been produced, TERMINATE.",
        })

    if not full_response and funcall_chains:
        try:
            args = {
                "model": model or _context.model,
                "input": input_items + [{
                    "role": "system",
                    "content": (
                        "[Constraints]\nYou have reached the maximum number of tool calls "
                        "for this turn. Do NOT call any more tools. Answer now using only "
                        "the information already gathered above, and say clearly if some "
                        "aspect could not be fully verified."
                    ),
                }],
            }
            if reasoning_effort and reasoning_effort != "none":
                args["reasoning"] = {"effort": reasoning_effort}
            forced = await asyncio.to_thread(state.openai_client.responses.create, **args)
            forced_text = (forced.output_text or "").strip()
            if forced_text:
                full_response = forced_text
                yield forced_text
        except Exception as e:
            import logging

            logging.warning("Forced final-answer completion (responses API) failed: %s", e)
