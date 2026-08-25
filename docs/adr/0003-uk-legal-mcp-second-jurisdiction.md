# ADR-0003: UK Legal MCP as a Second Jurisdiction, Full Tool Surface

## Status
Accepted

## Context

`[legal ai]` today is PH-only, backed solely by the juris.ph MCP (ADR-0002). There is no
jurisdiction concept anywhere in the codebase — persona routing is a flat prefix match in
`process_persona()`. Adding UK legal research means wiring in a second MCP server,
`https://uk-legal-mcp.fly.dev/mcp` — a different Streamable HTTP JSON-RPC server (stateless,
no auth, no Cloudflare UA spoofing needed) exposing 34 tools across case law, legislation,
Parliament/Hansard, bills, votes, committees, citations, and HMRC guidance — far broader than
juris.ph's 4 tools.

Domain terms for this change live in `CONTEXT.md` under **UK Legal search**.

## Decision

1. **Jurisdiction routing** — new persona tag `[legal ai uk]`, parallel to `[legal ai]`, not a
   parameter on the existing persona. PH and UK are fully separate personas with separate tool
   whitelists and separate prompt files.
2. **Tool surface** — expose all 34 UK MCP tools to `[legal ai uk]`, not a curated core subset
   (case law + legislation + citations only). This is deliberately broader than ADR-0002's
   narrow 4-tool PH surface — a future reader should know this was a considered choice, not
   scope creep.
3. **Manifest fidelity** — hand-condense all 34 tool entries into
   `resources/functions/user_functions.manifest`, matching the existing juris.ph entries' style
   (short description, flat params, no `outputSchema`/`anyOf` noise). The UK server's native
   schemas run ~13,000 tokens (name + description + inputSchema only, before `outputSchema`)
   versus a low-hundreds-of-tokens PH whitelist; condensing keeps per-turn tool-definition
   overhead manageable while preserving load-bearing correctness caveats (e.g. extent-checking,
   citation-before-quote, honorific-suffix gotchas) as short clauses.
4. **Client architecture** — extract a shared generic MCP client (`mcp_client.py`) from the
   juris.ph-specific one, since both servers speak the same stateless Streamable HTTP JSON-RPC
   `tools/call` protocol. `juris_mcp/client.py` keeps its Cloudflare UA spoofing as a
   server-specific option; a new `uk_legal_mcp/client.py` uses the shared base with no spoofing.

## Alternatives Considered

**Jurisdiction as a parameter on `[legal ai]`**

One persona; a jurisdiction field routes to PH vs UK tools/prompt.

*Why rejected:* no session/jurisdiction field exists yet in this codebase. A new persona tag is
far cheaper to ship, and the calling client already knows which country its user is in.

**Curated core tool subset (case law + legislation + citations, 12 tools)**

Mirrors PH's narrow-surface precedent from ADR-0002.

*Why rejected:* full 34-tool surface was wanted from the start, including
Parliament/Hansard/bills/votes/committees/HMRC — a materially larger product surface than
"answer a legal question," accepted deliberately rather than deferred to a phase 2.

**Pass through native UK tool schemas near-verbatim**

Preserves every upstream correctness caveat exactly as written.

*Why rejected:* ~13K tokens of tool definitions on every single turn is a heavy fixed tax on a
chat product. The existing PH manifest already proves hand-condensing works without losing the
correctness-critical caveats, as long as they're compressed rather than dropped.

**Duplicate a second standalone client module (no shared base)**

Matches ADR-0002's "one client per server" precedent literally.

*Why rejected:* with two MCP servers on the identical protocol, duplicating retry/unwrap
transport logic risks the two copies drifting apart. A shared base keeps transport-level fixes
in one place.

## Consequences

- `CONTEXT.md` gets a new **UK Legal search** section parallel to the existing **Legal search**
  section — not an edit-in-place, since the PH section's terms (In-Scope Legal Material, Legal
  Tool, Legal Citation) are worded PH-exclusively.
- `legal_citations.py`'s `collect_tool_result_urls()` must be extended to recognize the UK tool
  results' URL field names (e.g. `uri` for case law search results, not just
  `url`/`document.url`/`document.pdf_url`) — otherwise the Cite Gate will strip legitimate UK
  citations as "unverified."
- `resources/prompts/legal_prompt_uk.txt` is a new file, not a shared/conditional prompt — it
  mirrors the memo-style structure of `legal_prompt.txt` but is adapted for OSCOLA citation
  format (via `citations_format_oscola`/`citations_resolve`) and the wider UK source set.
- Per-turn token cost for `[legal ai uk]` is meaningfully higher than `[legal ai]` even after
  condensing (34 tools vs 4). Worth monitoring if latency/cost becomes an issue; trimming to the
  core subset remains an option later if it does.
