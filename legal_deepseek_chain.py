# -*- coding: utf-8 -*-
"""Legal-persona tool-calling loop on DeepSeek, kept fully separate from
legal_responses_chain.py (OpenAI /v1/responses, gpt-5.6-terra) and from
the_server.run_function_chain/streaming_run_function_chain (shared Chat
Completions loop used by every other persona).

Why a separate module instead of making the shared loop provider-aware:
run_function_chain/streaming_run_function_chain are not legal-exclusive — every
persona (cosmetics, garments, outfits, general chat, maps, nav, stylist,
tailor) funnels through them via a single state.openai_client. Threading
DeepSeek client selection into that shared code risks regressing every other
persona. Duplicating the loop here instead means gpt-5.6-terra's Responses-API
path and the shared Chat Completions path are both completely untouched — this
module is only ever reached when the legal persona's resolved LEGAL_CHAT_MODEL
is a DeepSeek model (see the_server._legal_model_override / the three dispatch
sites in reason_loop / streaming_reason_loop / the HITL-resume handler).

Structurally this is the same OpenAI-SDK Chat Completions loop as
the_server.run_function_chain, since DeepSeek's hosted API is OpenAI-SDK
compatible via base_url (confirmed in earlier sessions) — the only real
differences are: (a) the client and request kwargs come from llm_provider
(so temperature/reasoning_effort get dropped correctly per model), and (b) a
truncated-empty safety check, since deepseek-v4-flash was confirmed live
(2026-08-20) to spend its whole max_tokens budget on a separate
`reasoning_content` field and return empty `content` with finish_reason="length"
under a small enough cap -- caught here via llm_provider.is_truncated_empty
rather than silently returning a blank turn.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

import llm_provider


def run_function_chain_deepseek(
    state,
    messages: list,
    max_chains: int = 12,
    session_id: str = None,
    tools: list = None,
    query: str = "",
    model: str = None,
    reasoning_effort: str = None,
    temperature: float = None,
    auto_approval: bool = False,
):
    """Legal-persona equivalent of the_server.run_function_chain(), routed to
    DeepSeek via llm_provider instead of state.openai_client. reasoning_effort
    is accepted for call-site signature symmetry but never sent -- DeepSeek has
    no equivalent parameter (llm_provider.normalize_request_kwargs drops it).
    """
    from the_server import (
        _context,
        _describe_tool_args,
        _summarize_tool_result,
        broadcast_trace,
        execute_function_call,
    )

    resolved_model = model or _context.model
    client = llm_provider.build_client(resolved_model)
    available_manifest = tools if tools is not None else _context.fun_manifest
    funcall_chains = []
    function_outputs = []
    full_response = ""
    last_tool = None
    tool_log = []
    state.last_turn_tool_log = tool_log

    def perform_chat(msgs):
        args = llm_provider.normalize_request_kwargs(
            resolved_model,
            {
                "model": resolved_model,
                "messages": msgs,
                "n": 1,
                "stream": True,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            },
            has_tools=bool(available_manifest),
        )
        if available_manifest:
            args["tools"] = available_manifest
            args["tool_choice"] = "auto"
        return client.chat.completions.create(**args)

    for _ in range(max_chains):
        function_call = {"name": None, "arguments": ""}
        if last_tool:
            _cycle_summary = f"The AI received results from '{last_tool}' and is deciding whether it has enough information to answer or needs to take another step."
        else:
            _cycle_summary = "The AI is working through the question, deciding whether it needs to use a tool or can answer directly."
        broadcast_trace(
            "cognition",
            f"Cycle {_ + 1} — reasoning over {len(messages)} messages (model: {resolved_model})",
            session_id,
            summary=_cycle_summary,
        )
        stream_resp = perform_chat(messages)
        last_response = ""
        _finish_reason = None
        _locked_tool_index = None

        for chunk in stream_resp:
            choice = chunk.choices[0]
            _finish_reason = choice.finish_reason or _finish_reason
            delta = choice.delta
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_index = getattr(tc, "index", 0) or 0
                    if _locked_tool_index is None:
                        _locked_tool_index = tc_index
                    if tc_index != _locked_tool_index:
                        # DeepSeek can emit multiple parallel tool calls in one
                        # turn despite parallel_tool_calls=False (confirmed
                        # live 2026-08-20: their argument fragments otherwise
                        # get concatenated into one invalid JSON blob, making
                        # every call fail). Only the first tool call's
                        # fragments are kept; the rest are ignored this cycle.
                        continue
                    if tc.function.name:
                        function_call["name"] = tc.function.name
                    if tc.function.arguments:
                        function_call["arguments"] += tc.function.arguments
            elif hasattr(delta, "function_call") and delta.function_call:
                fc = delta.function_call
                if fc.name:
                    function_call["name"] = fc.name
                if fc.arguments:
                    function_call["arguments"] += fc.arguments
            elif hasattr(delta, "content") and delta.content:
                last_response += delta.content.replace("~", "-")

        if not function_call["name"] and not last_response.strip() and _finish_reason == "length":
            logging.warning(
                "[legal_deepseek_chain] Model '%s' hit its token limit while reasoning and "
                "produced no output this cycle (session=%s).", resolved_model, session_id,
            )
            broadcast_trace("cognition", "Model produced no output before hitting its token limit.", session_id,
                summary="The AI spent its response budget on internal reasoning and did not produce a visible answer this step.")
            break

        if not function_call["name"] and last_response:
            preview = last_response[:200].replace('\n', ' ')
            broadcast_trace("cognition", f"LLM produced final text ({len(last_response)} chars): \"{preview}{'…' if len(last_response) > 200 else ''}\"", session_id,
                summary="The AI has finished reasoning and is ready to deliver its response.")

        _xai_reason = None
        if last_response:
            _clean_lines = []
            for _ln in last_response.split("\n"):
                if _ln.strip().startswith("REASON:") and _xai_reason is None:
                    _xai_reason = _ln.strip()[7:].strip()
                else:
                    _clean_lines.append(_ln)
            last_response = "\n".join(_clean_lines).strip()

        last_response = llm_provider.strip_think_tags(last_response)

        if _xai_reason and function_call["name"]:
            broadcast_trace("cognition", f"Reasoning: {_xai_reason}", session_id,
                summary=f"In the AI's own words, it explained its decision: \"{_xai_reason}\"")

        if function_call["name"]:
            _tool_desc = next((t['function'].get('description', '') for t in available_manifest if t['function']['name'] == function_call['name']), '')
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

        if not function_call["name"]:
            # Only a tool-free turn counts as the real answer -- a turn that
            # narrates ("I will search for...") *and* calls a tool in the same
            # breath must not squat on full_response, or the max_chains
            # exhaustion fallback below never fires (confirmed live 2026-08-20:
            # deepseek-v4-flash narrates alongside its first tool call; gpt-5.6-terra
            # does not, which is why this never surfaced on that path).
            if last_response.strip():
                full_response = last_response.strip()
            break

        # HITL gate
        if not (_context.manual_auto_approval or auto_approval):
            return {"__hitl__": True, "function_call": function_call, "messages": messages, "tools": tools}

        try:
            cur_args = json.loads(function_call["arguments"])
        except Exception:
            cur_args = function_call["arguments"]

        is_dup = any(
            fc["name"] == function_call["name"] and fc["args"] == cur_args
            for fc in funcall_chains
        )
        if is_dup:
            broadcast_trace("control", f"BLOCKED duplicate call: `{function_call['name']}` — injecting memory reminder", session_id,
                summary=f"Safety check: The AI proposed to use '{function_call['name']}' again with the same inputs. This was blocked to prevent a redundant loop.")
            messages.append({"role": "system", "content": f"Function `{function_call['name']}` already called with same args. Do not repeat."})
            continue

        broadcast_trace("control", f"APPROVED: `{function_call['name']}` — no prior identical call found", session_id,
            summary=f"Safety check passed. The AI's proposed action is new — it has not taken this exact step before. Proceeding to execute '{function_call['name']}'.")
        broadcast_trace("action", f"Executing `{function_call['name']}`...", session_id,
            summary=f"The AI is now running '{function_call['name']}' to retrieve the information it needs.")

        funcall_chains.append({"name": function_call["name"], "args": cur_args})
        result = execute_function_call(function_call, session_id=session_id)
        if result is None:
            continue
        function_outputs.append((function_call["name"], result))
        last_tool = function_call["name"]
        state.turn_tool_calls = getattr(state, "turn_tool_calls", 0) + 1

        try:
            _rp = json.dumps(result, ensure_ascii=False)
        except Exception:
            _rp = str(result)
        _ctx = _summarize_tool_result(function_call["name"], result)
        tool_log.append({"name": function_call["name"], "args": cur_args, "summary": _ctx})
        broadcast_trace("action", f"Result from `{function_call['name']}`:\n{_rp[:300]}", session_id,
            summary=f"'{function_call['name']}' completed. {_ctx}")
        broadcast_trace("memory", f"Fact stored: `{function_call['name']}` result is now confirmed knowledge.\nValue: {_rp[:150]}", session_id,
            summary=f"The AI stored the result from '{function_call['name']}'. This confirmed knowledge will be used when composing the final response.")

        try:
            content = json.dumps(result, ensure_ascii=False)
        except Exception:
            content = str(result)
        messages.append({
            "role": "system",
            "content": (
                f"[Memory Fact]\nFunction `{function_call['name']}` returned:\n{content}\n\n"
                "Use this fact in all future reasoning. Do NOT re-call with the exact same arguments."
            ),
        })
        messages.append({
            "role": "system",
            "content": (
                "[Constraints]\nIf a complete response has been produced, TERMINATE. "
                "Only execute new actions if their conditions are fully satisfied."
            ),
        })

    if not full_response and funcall_chains:
        try:
            forced_kwargs = llm_provider.normalize_request_kwargs(
                resolved_model,
                {
                    "model": resolved_model,
                    "messages": messages + [{
                        "role": "system",
                        "content": (
                            "[Constraints]\nYou have reached the maximum number of tool calls "
                            "for this turn. Do NOT call any more tools. Answer now using only "
                            "the information already gathered above, and say clearly if some "
                            "aspect could not be fully verified."
                        ),
                    }],
                    "n": 1,
                    "temperature": temperature,
                },
            )
            forced = client.chat.completions.create(**forced_kwargs)
            if not llm_provider.is_truncated_empty(forced):
                full_response = llm_provider.strip_think_tags((forced.choices[0].message.content or "").strip())
        except Exception as e:
            logging.warning("[legal_deepseek_chain] Forced final-answer completion failed: %s", e)

    return full_response


async def _astream_llm(perform_chat_fn, messages):
    loop = asyncio.get_event_loop()
    q = asyncio.Queue()

    def _run():
        try:
            for chunk in perform_chat_fn(messages):
                loop.call_soon_threadsafe(q.put_nowait, chunk)
        except Exception as e:
            loop.call_soon_threadsafe(q.put_nowait, e)
        finally:
            loop.call_soon_threadsafe(q.put_nowait, None)

    import threading
    threading.Thread(target=_run, daemon=True).start()

    while True:
        item = await q.get()
        if item is None:
            break
        if isinstance(item, Exception):
            raise item
        yield item


async def streaming_run_function_chain_deepseek(
    state,
    messages: list,
    max_chains: int = 12,
    session_id: str = None,
    tools: list = None,
    query: str = "",
    model: str = None,
    reasoning_effort: str = None,
    temperature: float = None,
    auto_approval: bool = False,
):
    """Streaming/async equivalent of run_function_chain_deepseek, mirroring
    the_server.streaming_run_function_chain's shape and event contract
    (broadcast_trace calls, __HITL__ sentinel) exactly, but against DeepSeek.
    """
    from the_server import (
        _context,
        _describe_tool_args,
        _summarize_tool_result,
        broadcast_trace,
        execute_function_call,
    )

    resolved_model = model or _context.model
    client = llm_provider.build_client(resolved_model)
    available_manifest = tools if tools is not None else _context.fun_manifest
    funcall_chains = []
    function_outputs = []
    full_response = ""
    last_tool = None
    tool_log = []
    state.last_turn_tool_log = tool_log

    def perform_chat(msgs):
        args = llm_provider.normalize_request_kwargs(
            resolved_model,
            {
                "model": resolved_model,
                "messages": msgs,
                "n": 1,
                "stream": True,
                "temperature": temperature,
                "reasoning_effort": reasoning_effort,
            },
            has_tools=bool(available_manifest),
        )
        if available_manifest:
            args["tools"] = available_manifest
            args["tool_choice"] = "auto"
        return client.chat.completions.create(**args)

    think_filter = llm_provider.make_think_tag_stream_filter()

    for iteration in range(max_chains):
        function_call = {"name": None, "arguments": ""}
        if last_tool:
            _cycle_summary = f"The AI received results from '{last_tool}' and is deciding whether it has enough information to answer or needs to take another step."
        else:
            _cycle_summary = "The AI is working through the question, deciding whether it needs to use a tool or can answer directly."
        broadcast_trace("cognition", f"Cycle {iteration + 1} — reasoning over {len(messages)} messages (model: {resolved_model})", session_id,
            summary=_cycle_summary)
        await asyncio.sleep(0)
        _xai_buffer = ""
        _xai_first_line_done = False
        _xai_reason = None
        _iter_start = time.time()
        last_response = ""
        _first_token_time = None
        _finish_reason = None
        _locked_tool_index = None

        async for chunk in _astream_llm(perform_chat, messages):
            choice = chunk.choices[0]
            _finish_reason = choice.finish_reason or _finish_reason
            delta = choice.delta
            if hasattr(delta, "tool_calls") and delta.tool_calls:
                for tc in delta.tool_calls:
                    tc_index = getattr(tc, "index", 0) or 0
                    if _locked_tool_index is None:
                        _locked_tool_index = tc_index
                    if tc_index != _locked_tool_index:
                        # See matching comment in run_function_chain_deepseek:
                        # only the first parallel tool call's fragments are kept.
                        continue
                    if tc.function.name:
                        function_call["name"] = tc.function.name
                    if tc.function.arguments:
                        function_call["arguments"] += tc.function.arguments
            elif hasattr(delta, "function_call") and delta.function_call:
                fc = delta.function_call
                if fc.name:
                    function_call["name"] = fc.name
                if fc.arguments:
                    function_call["arguments"] += fc.arguments
            elif hasattr(delta, "content") and delta.content:
                if _first_token_time is None:
                    _first_token_time = time.time()
                part = think_filter(delta.content.replace("~", "-"))
                if not part:
                    continue
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

        if not function_call["name"] and not last_response.strip() and _finish_reason == "length":
            logging.warning(
                "[legal_deepseek_chain] Model '%s' hit its token limit while reasoning and "
                "produced no output this cycle (session=%s).", resolved_model, session_id,
            )
            broadcast_trace("cognition", "Model produced no output before hitting its token limit.", session_id,
                summary="The AI spent its response budget on internal reasoning and did not produce a visible answer this step.")
            break

        _iter_elapsed = time.time() - _iter_start
        if function_call["name"]:
            logging.info(
                "chain[%d] LLM→tool=%s llm=%.2fs session=%s (deepseek)",
                iteration, function_call["name"], _iter_elapsed, session_id,
            )
        else:
            ttft_iter = (_first_token_time - _iter_start) if _first_token_time else 0
            logging.info(
                "chain[%d] LLM→text chars=%d ttft=%.2fs total=%.2fs session=%s (deepseek)",
                iteration, len(last_response), ttft_iter, _iter_elapsed, session_id,
            )

        if not function_call["name"] and last_response:
            preview = last_response[:200].replace('\n', ' ')
            broadcast_trace("cognition", f"LLM produced final text ({len(last_response)} chars): \"{preview}{'…' if len(last_response) > 200 else ''}\"", session_id,
                summary="The AI has finished reasoning and is ready to deliver its response.")
            await asyncio.sleep(0)

        if _xai_reason and function_call["name"]:
            broadcast_trace("cognition", f"Reasoning: {_xai_reason}", session_id,
                summary=f"In the AI's own words, it explained its decision: \"{_xai_reason}\"")
            await asyncio.sleep(0)

        if function_call["name"]:
            _tool_desc = next((t['function'].get('description', '') for t in available_manifest if t['function']['name'] == function_call['name']), '')
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

        if not function_call["name"]:
            # See matching comment in run_function_chain_deepseek: only a
            # tool-free turn's text counts as the real answer, or narration
            # alongside a tool call squats on full_response and the max_chains
            # exhaustion fallback below never fires.
            if last_response.strip():
                full_response = last_response.strip()
            break

        # HITL gate: emit pending_approval event and stop streaming
        if not (_context.manual_auto_approval or auto_approval):
            yield f"__HITL__{json.dumps({'function_call': function_call, 'messages': messages, 'tools': [t['function']['name'] for t in (tools or [])]})}"
            return

        try:
            cur_args = json.loads(function_call["arguments"])
        except Exception:
            cur_args = function_call["arguments"]

        is_dup = any(fc["name"] == function_call["name"] and fc["args"] == cur_args for fc in funcall_chains)
        if is_dup:
            broadcast_trace("control", f"BLOCKED duplicate call: `{function_call['name']}` — injecting memory reminder", session_id,
                summary=f"Safety check: The AI proposed to use '{function_call['name']}' again with the same inputs. This was blocked to prevent a redundant loop.")
            await asyncio.sleep(0)
            messages.append({"role": "system", "content": f"Function `{function_call['name']}` already called. Do not repeat."})
            continue

        broadcast_trace("control", f"APPROVED: `{function_call['name']}` — no prior identical call found", session_id,
            summary=f"Safety check passed. The AI's proposed action is new — it has not taken this exact step before. Proceeding to execute '{function_call['name']}'.")
        await asyncio.sleep(0)
        broadcast_trace("action", f"Executing `{function_call['name']}`...", session_id,
            summary=f"The AI is now running '{function_call['name']}' to retrieve the information it needs.")
        await asyncio.sleep(0)

        funcall_chains.append({"name": function_call["name"], "args": cur_args})

        _tool_start = time.time()
        result = await asyncio.to_thread(execute_function_call, function_call, session_id=session_id)
        logging.info(
            "chain[%d] tool=%s exec=%.2fs session=%s (deepseek)",
            iteration, function_call["name"], time.time() - _tool_start, session_id,
        )
        if result is None:
            continue
        function_outputs.append((function_call["name"], result))
        last_tool = function_call["name"]
        state.turn_tool_calls = getattr(state, "turn_tool_calls", 0) + 1

        try:
            _rp = json.dumps(result, ensure_ascii=False)
        except Exception:
            _rp = str(result)
        _ctx = _summarize_tool_result(function_call["name"], result)
        tool_log.append({"name": function_call["name"], "args": cur_args, "summary": _ctx})
        broadcast_trace("action", f"Result from `{function_call['name']}`:\n{_rp[:300]}", session_id,
            summary=f"'{function_call['name']}' completed. {_ctx}")
        await asyncio.sleep(0)
        broadcast_trace("memory", f"Fact stored: `{function_call['name']}` result is now confirmed knowledge.\nValue: {_rp[:150]}", session_id,
            summary=f"The AI stored the result from '{function_call['name']}'. This confirmed knowledge will be used when composing the final response.")
        await asyncio.sleep(0)

        try:
            content = json.dumps(result, ensure_ascii=False)
        except Exception:
            content = str(result)
        messages.append({
            "role": "system",
            "content": f"[Memory Fact]\nFunction `{function_call['name']}` returned:\n{content}\n\nUse this fact. Do NOT re-call with the exact same arguments.",
        })
        messages.append({
            "role": "system",
            "content": "[Constraints]\nIf a complete response has been produced, TERMINATE.",
        })

    if not full_response and funcall_chains:
        try:
            forced_kwargs = llm_provider.normalize_request_kwargs(
                resolved_model,
                {
                    "model": resolved_model,
                    "messages": messages + [{
                        "role": "system",
                        "content": (
                            "[Constraints]\nYou have reached the maximum number of tool calls "
                            "for this turn. Do NOT call any more tools. Answer now using only "
                            "the information already gathered above, and say clearly if some "
                            "aspect could not be fully verified."
                        ),
                    }],
                    "n": 1,
                    "temperature": temperature,
                },
            )
            forced = await asyncio.to_thread(client.chat.completions.create, **forced_kwargs)
            if not llm_provider.is_truncated_empty(forced):
                forced_text = llm_provider.strip_think_tags((forced.choices[0].message.content or "").strip())
                if forced_text:
                    full_response = forced_text
                    yield forced_text
        except Exception as e:
            logging.warning("[legal_deepseek_chain] Forced final-answer completion failed: %s", e)
