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

## Legal search

**Legal Source**
The sole live authority for legal retrieval in this product: the juris.ph MCP (`search_jurisprudence`, `search_republic_acts`, `get_case`, `get_republic_act`). Local Postgres/pgvector RAG is not used for live search.
_Avoid_: hybrid RAG+MCP, dual corpus, Anycase DB for answers

**In-Scope Legal Material**
Philippine Supreme Court decisions and Republic Acts only. Requests about issuances, IRRs, ordinances, or other non-RA instruments are out of scope and must be declined clearly — not answered from model memory as if sourced.
_Avoid_: Anycase issuance corpus, general “Philippine law” without an MCP hit

**Legal Citation**
A user-facing reference to a retrieved document, expressed as the juris.ph shareable page URL (optionally paired with the official PDF link). Not a local `/sources/{id}` library path.
_Avoid_: numeric DB item_id, LEGAL_LIBRARY_URL sources links

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
