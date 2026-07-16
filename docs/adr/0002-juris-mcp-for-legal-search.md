# ADR-0002: juris.ph MCP as Sole Live Legal Source

## Status
Accepted

## Context

`[legal ai]` today retrieves from a local Anycase → Postgres/pgvector hybrid RAG stack
(`legal_rag/`, chat tools `search_legal` / `summarize_legal_case`, and HTTP surfaces under
`/api/legal/*` and `/legal/*`). Citations use numeric DB ids via `/sources/{id}`.

We want live legal retrieval to come from the public juris.ph MCP at `https://juris.ph/mcp`
instead of querying that database. The MCP exposes semantic search and fetch over Philippine
Supreme Court decisions and Republic Acts (lawphil-sourced; digests are AI-generated and may
err). It speaks Streamable HTTP JSON-RPC, is stateless, and requires no API key.

For a clean MCP-only test of the product path, the local `legal_rag/` stack is removed rather
than left as an offline archive. Rollback is via git history if needed.

Domain terms for this change live in `CONTEXT.md` under **Legal search**.

## Decision

**Replace live legal retrieval with juris.ph MCP (CHOSEN).**

1. **Legal Source** — juris.ph MCP is the sole live authority. Remove the local `legal_rag/`
   package, related live routes/config, and any chat wiring to Postgres/pgvector legal
   search. No local Legal Archive in-tree for this iteration (testing/clean break; recover
   from git if needed).
2. **In-Scope Legal Material** — Supreme Court decisions and Republic Acts only. Issuances,
   IRRs, ordinances, and other non-RA instruments are declined clearly — not answered from
   model memory as if sourced.
3. **Legal Tools** — expose the four MCP operations directly to the `[legal ai]` persona:
   `search_jurisprudence`, `search_republic_acts`, `get_case`, `get_republic_act`. Drop the
   `search_legal` / `summarize_legal_case` facade.
4. **Legal Citation** — user-facing links are juris.ph page URLs (optional official PDF).
   Drop `/sources/{id}` and `LEGAL_LIBRARY_URL`-based library links for live answers.
5. **Juris MCP Client** — in-process Python client calling `https://juris.ph/mcp` over
   Streamable HTTP. No `npx mcp-remote` sidecar; no Cursor-only wiring for product chat.
6. **Legal HTTP API** — remove public `/api/legal/*` and `/legal/*` search/ask/case/ingest
   surfaces. Live retrieval is chat Legal Tools only.
7. **Failures** — on MCP transport/client error, retry once then fail soft. On empty search
   (**Legal Retrieval Miss**), say nothing on-point was found. Never fall back to a local
   corpus; never invent holdings.
8. **Holding Statement** — before stating what a case or RA decides, the model must have
   called `get_case` / `get_republic_act` for that document (structured record). Use
   `include_full_text` only when quoting exact language or the user asks for the full
   document. Search digests alone are insufficient.
9. **Legal Content Use** — non-commercial / internal/research posture with attribution via
   Legal Citations. Commercial redistribution is out of scope for this decision.

**Chosen because:**
- One live source of truth avoids dual citation ids and hybrid failure modes
- MCP’s case/RA boundary matches the product’s narrowed promise better than a local
  “jurisprudence / law / issuance” mapper
- Direct HTTP MCP fits a Python API without a Node/`npx` dependency
- Removing `legal_rag/` for this test avoids an unwired Archive that could be reconnected
  by mistake; git is enough to restore if the experiment fails

## Alternatives Considered

**Hybrid MCP + local RAG**

Try juris.ph first; fall back to Postgres when MCP misses or for issuances/non-RA law.

*Why rejected:* Dual citation identities, unclear “what counts as a source,” and reintroduces
the coverage lie that Anycase issuances are still first-class when the product promise is
SC + RAs.

**Chat-only MCP; keep HTTP legal APIs on RAG**

Wire MCP into `[legal ai]` tools but leave `/api/legal/*` on pgvector.

*Why rejected:* Splits product behavior; clients and chat would disagree on sources and
coverage. Full replacement of live retrieval was preferred; HTTP legal APIs are removed
rather than rewritten.

**Keep `search_legal` facade over MCP**

One umbrella tool that routes to jurisprudence vs RA search internally.

*Why rejected:* Hides MCP’s real boundaries and recreates a brittle content-type mapper.
Exposing the four tools lets the model choose deliberately.

**`npx mcp-remote` stdio sidecar**

Bridge stdio MCP from a Node process into Python.

*Why rejected:* Extra runtime, cold starts, and process management. juris.ph already speaks
Streamable HTTP; an in-process HTTP MCP client is enough.

**Keep local RAG as an offline Legal Archive**

Leave `legal_rag/`, ingest, and DB in the repo but never query them from chat.

*Why rejected for this iteration:* Extra surface area and risk of accidental rewiring during
MCP-only testing. Prefer a clean break; restore from git if an Archive is needed later.

**Answer from search digests without `get_*`**

Lower latency; skip structured fetch for holdings.

*Why rejected:* juris.ph digests are AI-generated and may be wrong. Holding Statements require
a structured `get_*` fetch; full text is reserved for quotes / explicit full-document asks.

## Consequences

- Persona whitelist and `resources/prompts/legal_prompt.txt` must target the four Legal Tools
  and juris.ph URL citations; drop numeric `/sources/{id}` repair against Postgres
- Implement a Juris MCP Client (Streamable HTTP, one retry on transport failure) and bind
  Legal Tools to it
- Remove `/api/legal/*` and `/legal/*` routes and delete `legal_rag/` (plus related env/config
  that only served local legal RAG)
- Out-of-scope legal topics and empty MCP searches become explicit declines / misses
- If commercial use is later required, licensing must be revisited before expanding
  Legal Content Use — this ADR does not settle a SaaS redistribution license
- If MCP-only testing fails, restore `legal_rag/` from git rather than maintaining a dormant
  in-tree Archive during the experiment
