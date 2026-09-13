# Domain Context — chat-wonder-v2-api

## Glossary

**Persona**
A named reasoning mode activated by a prefix in `user_input` (e.g. `[stylist]`, `[legal ai]`, `[garment]`). Each persona has a whitelisted tool set and an addendum that overrides the system prompt. Personas are resolved by `process_persona()` in `the_server.py`.

**Miraj (Stylist persona)**
The fashion-AI persona. Triggered by the `[stylist]` prefix. Handles outfit recommendations, cosmetics, maps, and tailor flows. Has two execution paths: a structured direct path (when a `category` field is present in the request) and an LLM-mediated path (when the input is natural language).

**Greater Category**
The top-level bucket a user selects in the fashion catalog UI. One of: `casual`, `formal`, `outdoor`. Not stored in the database — used only as a grouping label for metacategories.

**Metacategory**
A style sub-classification under a Greater Category. Passed as a comma-separated string matching the `metaCategory` filter on the outfits API.
- Casual: Streetwear, Athleisure, Vintage, Minimalist, AvantGarde, Traditional, Cultural
- Formal: Business, SmartCasual, Luxury, Uniform
- Outdoor: Winterwear, Summerwear, Rainwear, Springwear, Autumnwear, Sportswear, Activewear

**Outfit**
A pre-composed set of garments stored in mirror-api. Identified by a UUID (`id`). Contains a hero image and a list of constituent Garments. chat-wonder-v2-api fetches Outfits from the external outfits API.

**Garment**
An individual clothing item belonging to an Outfit. Has a UUID, fitting slot, garment type, layer level, silhouette, and gender.

**Structured Outfit Search**
The direct execution path for `[stylist]` requests that carry a `category` field. Bypasses the Miraj LLM persona loop. Calls `search_outfits_by_category()` directly, which filters the outfit catalogue by metacategory + gender, runs an inner LLM selection, and returns only outfit UUIDs. The calling system (mirror-api) fetches full outfit data for those UUIDs independently.

**`[OUTFIT_IDS]` block**
A WebSocket emission block containing a JSON array of outfit UUID strings. Emitted by the structured outfit search path. Distinct from `[GARMENT_DATA]`, which carries full hydrated outfit objects and is emitted by the LLM-mediated path.

**SCL Tracer**
A background thread that fires a lightweight duplicate call after certain tool executions, for XAI compliance logging. Fires after `search_outfits_by_category` (structured path) and after `get_outfits_by_category` (LLM path).

## Legal search (Philippines)

**Legal Source**
The sole live authority for legal retrieval in this product: the juris.ph MCP (`search_jurisprudence`, `search_republic_acts`, `get_case`, `get_republic_act`). Local Postgres/pgvector RAG is not used for live search.
_Avoid_: hybrid RAG+MCP, dual corpus, Anycase DB for answers

**In-Scope Legal Material**
Philippine Supreme Court decisions and Republic Acts only. Requests about issuances, IRRs, ordinances, or other non-RA instruments are out of scope and must be declined clearly — not answered from model memory as if sourced.
_Avoid_: Anycase issuance corpus, general “Philippine law” without an MCP hit

**Legal Citation**
A user-facing reference to a retrieved document, expressed as the juris.ph shareable page URL (optionally paired with the official PDF link). Not a local `/sources/{id}` library path.
_Avoid_: numeric DB item_id, LEGAL_LIBRARY_URL sources links

**Legal Result Pool**
`state.last_search_legal_results`: every row/record a legal tool returned this session, for both tenants. Always accumulated (search rows deduped by id; get_* entries appended), never overwritten, bounded by `LEGAL_POOL_MAX`. It is the allowed set for the Cite Gate and the corpus for quote verification. Confirmed live: overwriting on each PH `search_*` call wiped vetted `get_case` entries and cost 17–20 legitimate citations per turn.
_Avoid_: `state.last_search_legal_results = result["results"]`; resetting the pool mid-turn

**Cite Gate**
Post-response filter in legal mode: markdown links whose href is not an exact URL from the current tool-result pool (including truncated `juris.ph/case/...` placeholders) are demoted to plain link text. Implemented in `legal_citations.py`.
_Avoid_: leaving phantom juris.ph hrefs in the final answer

**Legal Analysis Protocol**
Always-on instructions injected for every `[legal ai]` fact-pattern turn: issue-spot the user's questions, run targeted multi-searches (including supporting case lines), fetch before holdings, and answer in an AnyCase-style memo — opening direct answer, sections mirroring (a)/(b)/(c) with nested practical steps (Why / Core requirements / Important note), authority trails with exact juris.ph URLs, direct-answer reprise, one clarifying question. Sample doctrines (divorce, floating status, OPC, free patent, cyber libel, etc.) are non-exhaustive hints — not a closed list of supported topics.
_Avoid_: hardcoding support to a fixed Q1–Q5 eval set only; thin FAQ answers that skip operational next steps when the user asked a multi-part fact pattern; inventing BIR/LGU issuances to mimic AnyCase depth

**Legal Tool**
One of the four juris.ph MCP operations exposed to the `[legal ai]` persona: `search_jurisprudence`, `search_republic_acts`, `get_case`, `get_republic_act`. The model chooses among these directly; there is no umbrella `search_legal` / `summarize_legal_case` facade.
_Avoid_: search_legal, summarize_legal_case, content_types mapper

**Juris MCP Client**
The in-process client the API uses to invoke juris.ph Legal Tools for the `[legal ai]` persona. There is no local legal RAG path.
_Avoid_: Cursor-only MCP wiring, local pgvector legal search

**Legal HTTP API**
Removed. Live legal retrieval is available only through the `[legal ai]` chat persona and its Legal Tools. No public `/api/legal/*` or `/legal/*` search/ask/case/ingest surface.
_Avoid_: /api/legal/search, /legal/ask, numeric case-by-id HTTP reads

**Legal Retrieval Miss**
A completed MCP search that returns no sufficiently relevant results. The persona must say nothing on-point was found — not invent holdings, not fall back to any local corpus.
_Avoid_: hallucinated case law, local RAG fallback on empty hits

**Holding Statement**
Any claim about what a case or RA decides or requires. Before making a Holding Statement the model must have called `get_case` or `get_republic_act` for that document (structured record; full text only when quoting exact language or the user asks for the full document). Search digests alone are not enough.
_Avoid_: citing from search digest only, always-on include_full_text

**Legal Content Use**
Live legal material from juris.ph / lawphil is used under a non-commercial (internal/research) posture with attribution via Legal Citations. Commercial redistribution is out of scope for this product decision.
_Avoid_: assuming SaaS commercial license is settled

## Legal search (UK)

**UK Legal Source**
The sole live authority for UK legal retrieval: the UK Legal MCP at `https://uk-legal-mcp.fly.dev/mcp`. A distinct persona and MCP server from the Philippines Legal Source — the two are never mixed in one turn.
_Avoid_: routing UK questions through juris.ph, jurisdiction auto-detection

**UK Legal Tool**
Any of the 34 operations exposed by the UK Legal MCP to the `[legal ai uk]` persona, spanning case law (`case_law_search`, `judgment_get_*`), legislation (`legislation_search`, `legislation_get_*`), citations (`citations_parse`, `citations_resolve`, `citations_network`, `citations_format_oscola`), Parliament/Hansard, bills, votes, committees, and HMRC guidance. Deliberately the full server surface, not a curated core — see ADR-0003.
_Avoid_: a narrowed subset matching the PH Legal Tool count; an umbrella facade tool

**UK Legal Citation**
A user-facing reference to a UK judgment, legislation section, or Hansard contribution, formatted per OSCOLA convention (via `citations_format_oscola`/`citations_resolve`), using the source's own public URL (e.g. National Archives judgment URI, legislation.gov.uk, hansard.parliament.uk).
_Avoid_: juris.ph-style plain URL citation for UK sources, inventing an OSCOLA citation without resolving it first

**UK Section URL**
The canonical `https://www.legislation.gov.uk/{type}/{year}/{number}/section/{n}` page. The UK Legal MCP returns no URL on `legislation_get_section` and a `/search?title=` page as `citations_resolve`'s `resolved_url` for sections, so `uk_legal_mcp/urls.py` synthesises the section URL from the call args / earlier search rows before the result reaches the model or the Legal Result Pool. Fetched sections are cited directly (`<Act> <year>, s <n>` → that `url`), reserving `citations_resolve`/`citations_format_oscola` for case law.
_Avoid_: linking section citations to a `/search?title=` page; spending 2 tool calls per already-fetched section

**UK MCP Client**
The shared generic MCP client (`mcp_client.py`) instantiated against the UK Legal MCP, with no Cloudflare UA spoofing (unlike the Juris MCP Client). See ADR-0003 for why the transport layer is shared with the PH client rather than duplicated.
_Avoid_: a second hand-copied Streamable HTTP client with its own retry/unwrap logic

## Legal answer verification (PH + UK)

**Whole-Case Preload**
ilovelawyer-api sends `case_document_texts` ([{id, name, text}] for every READY document) when the case's extracted text fits its `CASE_FULL_TEXT_INLINE_CHARS` (200k chars); `sync_active_case_documents` seeds them into the document cache as full, non-partial entries so the model starts the turn with every exhibit in context. `CASE_DOCUMENT_TOKEN_BUDGET` was raised to 60k tokens to hold one. Brackenmoor before/after: 18 single-document fetches per question → 0.
_Avoid_: evicting preloaded documents to fit a smaller budget; fetching documents one at a time when the case fits

**Partial Document Read**
A `get_case_document` result produced with `case_document_chunk_ids`, `offset/limit`, or the char cap is marked `partial: true` with a `note` telling the model how to get the whole document; the cache never lets a partial read replace a full one, never serves a cached partial to a whole-document request, and the [CASE FILE] manifest marks partial entries. Both legal prompts require any document the question cites by paragraph/part/item to be read in full first. Confirmed live: the model previously treated a relevance-filtered slice as the exhibit (D01 §25, D20.3).
_Avoid_: treating a filtered fetch as "the document"; serving the cache for a re-read request

**Sub-question Budget**
`_legal_model_override` adds `LEGAL_CHAINS_PER_SUBQUESTION` (6) tool calls per numbered/lettered part beyond the first (`_count_sub_questions`), ceiling `LEGAL_MAX_CHAINS_CEILING` (80). A four-part assessment question exhausted the old 60-call cap.

**UK Doctrine Guards**
`append_uk_doctrine_guards` (legal_fact_boost.py) — the England & Wales counterpart of the PH guards, selected by `jurisdiction` in `_finalize_legal_response`: whistleblowing burden/remedies without qualifying service, DPA 2018 Part 3 vs Part 2, CJA 2003 s.117 business records, Northern Ireland extent. PH guards no longer run on UK answers.
_Avoid_: running PH guards on UK answers (Civil Code Art. 33 on a Leeds libel question)

**Verifier Feedback Loop**
Inside the tool-calling loop, when the legal model stops calling tools and produces a draft, the same checks the Cite Gate and quote-strip run post-hoc are run on the draft (`legal_verify.audit_legal_draft`). If anything would be demoted, the draft plus a `[VERIFIER …]` system message (what failed, how to fix it, the retrieved title→URL pool) is appended and the model revises. Bounded by `LEGAL_VERIFY_MAX_ROUNDS` (default 1; `0` disables — rollback lever). The post-hoc finalizer stays as the safety net for whatever survives. In the streaming path the chain yields `__DRAFT_DISCARD__` so `chat_stream` drops the buffered draft; a `self_check` `[TRACE]` step shows the revision in the research-step list.
_Avoid_: silently gating a citation the model was never told about; unbounded revise rounds; running the loop for non-legal personas; streaming legal draft text live (the discard only works because legal text is buffered server-side)
