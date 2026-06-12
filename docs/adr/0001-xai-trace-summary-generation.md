# ADR-0001: XAI Trace Event Summary Generation Strategy

## Status
Accepted

## Context

Each R-CCAM trace event broadcast by `broadcast_trace()` carries a `text` field that is
technically precise but written for developers. To satisfy XAI compliance requirements and
serve non-technical audiences (lawyers, clients, business stakeholders, compliance reviewers),
each event needs a second `summary` field in plain English.

Three strategies were evaluated for generating this `summary`.

## Decision

**Option 1 — Rule-based templates (CHOSEN)**

Pre-written plain-English strings with variable interpolation, defined in `the_server.py`
alongside each `broadcast_trace` call. The LLM's own `REASON:` line (already emitted as a
genuine chain-of-thought artifact before each tool call) is reused as the summary for
tool-proposal events — it is the most natural and legally authentic explanation available.

**Chosen because:**
- Zero latency and zero API cost — runs on the hot path of every request
- Deterministic — same event always produces the same explanation shape
- The REASON line already provides genuine natural-language reasoning for the most important
  step (tool selection); templates cover the rest adequately
- Sufficient for EU AI Act XAI compliance: explanations must be genuine, not necessarily
  generative

## Alternatives Considered

**Option 2 — LLM-generated summaries**

A secondary LLM call (e.g. `gpt-4o-mini`) fires for each trace event and produces a
contextually rich plain-English explanation of that step.

*Why rejected:* Adds latency and API cost to every trace event on the hot path. Trace events
fire multiple times per request (one per R-CCAM stage); the cost compounds quickly at scale.
More importantly, a generated explanation of a decision is still a reconstruction — it does
not strengthen the XAI compliance argument the way a genuine chain-of-thought REASON line does.

**Option 3 — Template-based with LLM fallback**

Templates handle common cases; an LLM call fills in only where templates cannot (e.g.
interpreting ambiguous tool arguments or unusual result shapes).

*Why rejected:* Adds architectural complexity (conditional LLM calls, fallback logic) for
marginal gain over Option 1. The REASON line already handles the hardest case (tool
selection intent) natively. Revisit if templates prove insufficient after real-world usage.

## Consequences

- `broadcast_trace()` gains an optional `summary: str = None` parameter
- The `summary` field is included in the SSE JSON payload when provided
- `run_glass_box_tracer.html` renders `summary` prominently and shows `text` in a
  collapsible "Technical detail" panel
- Any future consumer of `/trace-stream` (mobile app, PDF report generator, compliance
  dashboard) inherits human-readable summaries automatically
- If LLM-generated summaries become a requirement (e.g. richer compliance reports),
  Option 2 can be introduced as a post-processing pass on stored events without changing
  the hot-path broadcast logic
