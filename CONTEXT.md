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
