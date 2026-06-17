# Handoff: mirror-scl-fixes

**Repo:** `chat-wonder-v2-api`
**Branch:** `staging-scl-changes`
**Date:** 2026-06-17
**Context:** Vivatech demo is imminent. The session focused on making the `[stylist]` persona (Miraj) route all outfit and cosmetics requests through the LLM so that SCL (Glass-Box Tracer) shows the full RCCAM loop instead of nothing.

---

## Background

Mirror sends structured requests to chat-wonder with a `[stylist]` prefix. Previously, when `category` or `skin_analysis` was present in the payload, the server **bypassed Miraj entirely** and called internal functions directly. This meant SCL showed no cognition, no tool calls — just a bare request event. The whole point of the RCCAM loop is to show Miraj thinking and calling tools.

The fix: remove all bypass blocks, inject structured data as annotations, let Miraj call the tools herself.

---

## What Changed This Session

### `resources/functions/user_functions.py`

- **`get_cosmetics_by_skin_type(skin_type, sets=6)`** — completely rewritten. Now calls `_fetch_cosmetics_for_profile(skin_upper.lower(), [])`, shuffles results, returns `{"success": True, "skin_type": ..., "ids": [...]}`. Previously only returned `{"success": True, "skin_type": ...}` with no IDs.

- **`get_outfits_by_category(category, gender, weather_json=None, sets=4)`** — added `sets` param. Now returns `filtered[:n_sets]` instead of hardcoded `[:4]`.

- **Hot-weather filter** — `_HOT_THRESHOLD_C = 20.0`, `_has_heavy_outerwear()`, `_parse_temp_c()` added. Outfits with `layerLevel == OUTER` are filtered when temp ≥ 20°C.

### `the_server.py` — HTTP `/chat` handler

- **Removed** all three bypass blocks (combined, outfit-only, cosmetics-only) — ~125 lines gone. These were `if persona == "stylist" and request.category ...` early-return blocks.

- **Added annotation injections** in the stylist section:
  - `[OUTFIT_CATEGORY:{first_item_of_meta}]` from `request.category.get("meta")`
  - `[SKIN_TYPE:{skinType}]` from `request.skin_analysis.get("skinType")`
  - `[USER_GENDER:{gender}]` from `request.gender or state.confirmed_gender`
  - `[OUTFIT_SETS:{n}]` from `request.fsets or request.sets` (capped 1–4)
  - `[COSMETICS_SETS:{n}]` from `request.csets or request.sets` (capped 1–6)

- **`execute_function_call`** — added `state.last_cosmetics_ids_result = result.get("ids", [])` when `func_name == "get_cosmetics_by_skin_type"` and result has IDs.

### `the_server.py` — Miraj system prompt

- Added **Category rule**: "if `[OUTFIT_CATEGORY:X]` is present, use X exactly as the category parameter"
- Added **Sets rules**: pass `[OUTFIT_SETS:N]` as `sets` to `get_outfits_by_category`; pass `[COSMETICS_SETS:N]` as `sets` to `get_cosmetics_by_skin_type`
- Added `[OUTFIT_CATEGORY:...]` and `[SKIN_TYPE:...]` to the NEVER mention list

### `resources/functions/user_functions.manifest`

- `get_outfits_by_category` — added `sets` (integer, optional) and updated description
- `get_cosmetics_by_skin_type` — added `sets` (integer, optional), updated description to say "Returns product IDs"

---

## Current Flow (HTTP REST)

### Outfits
```
Mirror POST /chat
{
  "user_input": "[stylist] I'm looking for casual outfits...",
  "session_id": "...",
  "gender": "MALE",
  "category": { "meta": "Casual,Streetwear,Athleisure,..." },
  "weather": { "temperature_2m": 28 },
  "fsets": 4
}

→ Server injects: [OUTFIT_SETS:4], [COSMETICS_SETS:6], [USER_GENDER:MALE],
                  [OUTFIT_CATEGORY:Casual], [FRONTEND_WEATHER:{...}]
→ Miraj calls: get_outfits_by_category(category="Casual", gender="MALE", sets=4, weather_json="...")
→ Function returns: {"success": true, "ids": ["uuid1", "uuid2", "uuid3", "uuid4"]}
→ state.last_outfit_ids_result = [ids]
→ REST response: { "response": "...", "outfit_ids": ["uuid1",...], "cosmetics_ids": null }
→ WS emits: [OUTFIT_IDS]["uuid1",...] → __END__
→ SCL shows: Request → Cognition → Action (tool call) → Result ✓
```

### Cosmetics
```
Mirror POST /chat
{
  "user_input": "[stylist] Recommend skincare for my skin type.",
  "session_id": "...",
  "skin_analysis": { "skinType": "DRY", "concerns": ["sensitivity"] },
  "weather": { "temperature_2m": 28 },
  "csets": 6
}

→ Server injects: [COSMETICS_SETS:6], [OUTFIT_SETS:4], [SKIN_TYPE:DRY],
                  [SKIN_ANALYSIS:{...}], [FRONTEND_WEATHER:{...}]
→ Miraj calls: get_cosmetics_by_skin_type(skin_type="DRY", sets=6)
→ Function returns: {"success": true, "skin_type": "DRY", "ids": ["id1",...]}
→ state.last_cosmetics_ids_result = [ids]
→ REST response: { "response": "...", "outfit_ids": null, "cosmetics_ids": ["id1",...] }
→ WS emits: [COSMETICS_IDS]["id1",...] → __END__
→ SCL shows: Request → Cognition → Action (tool call) → Result ✓
```

---

## What Is Still Broken / Pending

### 1. WS bypass blocks still exist
The WebSocket handler (`/chat-stream`) has its own three bypass blocks (combined, outfit-only, cosmetics-only) starting around line 2327. **These were NOT removed.** The WS path still bypasses Miraj. Mirror-app appears to use REST, so this may not affect the demo — but the WS button in the HTML testers will still bypass.

To fix: remove the three `if persona == "stylist" and data.get("category")...` blocks from the WS handler, and add the same annotation injections that the REST handler now has (see the WS stylist section around line 2339–2380).

### 2. HTML testers need updating
- `new-outfit-fetching.html` — subtitle still says "bypasses Miraj LLM". `buildPayload()` sends `user_input: "[stylist]"` (works due to annotations but misleading). REST button works; WS button still bypasses.
- `new-cosmetics-search.html` — same subtitle issue. REST now returns `cosmetics_ids` correctly. WS still bypasses.

Update subtitles to: "Routes through Miraj LLM — returns IDs via tool call."
Optionally update `user_input` in `buildPayload()` to be a real sentence for clarity.

### 3. Combined payload (both category + skin_analysis) — HTTP REST
The old combined bypass sent both outfit and cosmetics in parallel. Now they go through Miraj separately. If mirror ever sends both `category` and `skin_analysis` in one request, Miraj will only handle whichever annotation she sees first. This edge case has not been tested.

---

## Files Modified
- `resources/functions/user_functions.py`
- `resources/functions/user_functions.manifest`
- `the_server.py`

## Files NOT Modified
- `mirror-api/` — do not touch
- `mirror-app/` — do not touch

---

## Suggested Skills

- `/diagnose` — if the WS path still isn't showing RCCAM loop after removing WS bypasses
- `/verify` — to confirm outfit_ids and cosmetics_ids appear in REST responses after deploy
- `/code-review` — sanity check the annotation injection ordering in the_server.py stylist section
