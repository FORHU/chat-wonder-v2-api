# Plan: Related Cases Tab — Powered by Ranked/Deduped Juris MCP Results

Status: Proposed (not yet implemented)
Spans repos: `chat-wonder-v2-api` (this repo), `ilovelawyer-api`, `ilovelawyer-app`

## Context

`app.ilovelawyer.com` has no working "related cases" surface today. The only prior
attempt — a `[RELATED_QUERIES]` prompt-injected tag asking the model to name 3-5
follow-up search terms — is dead: nothing parses it server-side, and the frontend
(`assistant-message.tsx` in ilovelawyer-app) explicitly regexes it out and throws it
away with a comment saying the feature "doesn't exist yet."

Separately, `chat-wonder-v2-api` already builds a real "active tool-result pool"
(`ChatState.last_search_legal_results`) every time the legal persona calls the four
Juris MCP tools (`search_jurisprudence`, `search_republic_acts`, `get_case`,
`get_republic_act`). That pool is already mapped to `source_metadata` and already sent
over the wire — as a REST field and as an ad hoc `"[Sources]" + json` WebSocket frame
sent right before the `__END__` sentinel. **ilovelawyer-api's WS client
(`chatWonder.ts`) already detects the `"[Sources]"` frame today and explicitly discards
it** (`sourcesDropped` flag) — so the data transport already exists end-to-end; it is
being thrown away, not missing.

Two real gaps stand in the way of "most relevant, not just a simple match":

1. **No ranking or dedup exists anywhere in chat-wonder-v2-api.** `search_*` calls
   *overwrite* the pool, `get_*` calls *append* unconditionally — there's no merge, no
   id-based dedup, and nothing sorts by the `score` field the MCP already returns per
   row (confirmed by full reads of `legal_citations.py`, `the_server.py`,
   `legal_fact_boost.py`). A raw dump of this pool would surface duplicates and
   unranked noise — literally "just a simple match."
2. **ilovelawyer-api/ilovelawyer-app don't consume it at all.** The WS client drops
   `[Sources]`, there's no Prisma column to persist related cases, no API field to
   return them, and the frontend has no Tabs primitive or message field to render them.

This plan closes both gaps: rank + dedup the pool once, close to the source of truth
(chat-wonder-v2-api, where the ADR-0002 tool-call discipline already lives), then wire
the already-flowing-but-discarded data through ilovelawyer-api into a real tab in
ilovelawyer-app.

---

## 1. chat-wonder-v2-api — rank, dedup, and emit `related_cases`

**New function** in `legal_citations.py` (next to `collect_tool_result_urls`):

```python
def select_related_cases(search_results: list, limit: int = 6) -> list:
    """Dedupe the legal tool-result pool by document identity and rank it.

    get_case/get_republic_act entries are model-vetted (ADR-0002 requires a
    structured fetch before any holding statement), so they outrank raw,
    unvetted search rows for the same document. Within each tier, sort by the
    MCP's own relevance score (search rows only; vetted entries have none and
    keep tier order).
    """
    def identity(row: dict) -> str:
        return (
            str(row.get("item_id") or row.get("id") or "")
            or str((row.get("metadata") or {}).get("id") or "")
            or (row.get("url") or (row.get("metadata") or {}).get("url") or "")
        )

    best: dict[str, dict] = {}
    for row in search_results or []:
        key = identity(row)
        if not key:
            continue
        vetted = "document" in row or row.get("prefetched")
        candidate = {**row, "_vetted": vetted}
        existing = best.get(key)
        if existing is None or (vetted and not existing["_vetted"]):
            best[key] = candidate

    ranked = sorted(
        best.values(),
        key=lambda r: (not r["_vetted"], -(r.get("score") or 0.0)),
    )

    return [
        {
            "type": r.get("type") or "legal_document",
            "title": r.get("title"),
            "url": r.get("url") or (r.get("metadata") or {}).get("url"),
            "case_number": r.get("case_number"),
            "ra_number": r.get("ra_number"),
            "year": r.get("year") or (r.get("metadata") or {}).get("year"),
            "snippet": r.get("snippet"),
            "relevance": r.get("score"),
            "vetted": r["_vetted"],
        }
        for r in ranked[:limit]
    ]
```

**Wire it in**, `the_server.py`, right where `source_metadata` is already produced:

- REST `/chat` (~line 1986, alongside `state.source_metadata` assignment): add
  ```python
  related_cases = legal_citations.select_related_cases(state.last_search_legal_results)
  ```
  and include `"related_cases": related_cases` in the response dict (~1983-1994).
- WS `/chat-stream` (~line 2450, same call site that emits `[Sources]` today): emit a
  **new, dedicated** frame right before `__END__` so existing `[Sources]` consumers are
  untouched:
  ```python
  await websocket.send_text(f"[RELATED_CASES]{json.dumps(related_cases)}")
  ```

**New ADR**: `chat-wonder-v2-api/docs/adr/0003-legal-related-cases-ranking.md` —
records that dedup/ranking didn't exist before, why vetted (`get_*`) entries outrank
raw search rows, and the new `related_cases` field / `[RELATED_CASES]` frame contract.

---

## 2. ilovelawyer-api — capture the frame, persist it, expose it

**`src/utils/chatWonder.ts`** — stop discarding, start capturing. Add a
`[RELATED_CASES]` parser alongside the existing `[Sources]`-drop handling, and change
`streamChatWonderMessage`'s resolution shape from a bare `string` to
`{ content: string; relatedCases: RelatedCase[] }`:

```ts
export interface RelatedCase {
  type: string;
  title: string | null;
  url: string | null;
  caseNumber: string | null;
  raNumber: string | null;
  year: unknown;
  snippet: string | null;
  relevance: number | null;
  vetted: boolean;
}

// inside ws.onmessage, alongside the existing "[Sources]" handling:
const relatedIdx = message.indexOf("[RELATED_CASES]");
if (relatedIdx !== -1) {
  try {
    relatedCases = JSON.parse(message.slice(relatedIdx + "[RELATED_CASES]".length));
  } catch { /* leave relatedCases as [] on malformed frame */ }
  if (relatedIdx > 0) {
    const clean = message.slice(0, relatedIdx);
    accumulated += clean;
    onChunk(clean);
  }
  return;
}
```

`finish()` resolves `{ content: accumulated, relatedCases }` instead of `accumulated`.

**`src/services/chat.service.ts` `sendMessage()`**:
- Change the Redis cache value from a raw string to
  `JSON.stringify({ content, relatedCases })` (still one key, same TTL) so a cache hit
  replays related cases too.
- After persisting the assistant message (mirroring the existing
  `if (timeline) await ChatRepo.saveTimeline(...)` / `saveMindMap` calls):
  ```ts
  if (relatedCases.length) {
    await ChatRepo.saveRelatedCases(assistantMessage.id, relatedCases);
  }
  ```

**Prisma migration** — new model, following the exact `MessageTimeline`/
`MessageMindMap` pattern (`prisma/schema.prisma`):
```prisma
model MessageRelatedCases {
  id        String   @id @default(uuid())
  messageId String   @unique
  items     Json
  createdAt DateTime @default(now())
  message   Message  @relation(fields: [messageId], references: [id], onDelete: Cascade)
}
```
Add `relatedCases MessageRelatedCases?` to `Message`, and `ChatRepo.saveRelatedCases`
next to `saveTimeline`/`saveMindMap` (`src/repositories/chat.repository.ts`).

**New endpoint** (no stream-protocol change needed — same pattern as
`legal-rag.controller.ts:getRelated`, keyed by conversation since the frontend already
knows `conversationId` without needing the new message's id):

`src/routes/chat.route.ts`:
```ts
router.get("/conversations/:conversationId/related-cases", asyncHandler(ChatCtrl.getRelatedCases));
```

`src/controllers/chat.controller.ts`:
```ts
static async getRelatedCases(req: Request, res: Response) {
  const { conversationId } = req.params;
  const relatedCases = await ChatSvc.getRelatedCases(req.user.userId, conversationId);
  return res.status(200).json({ relatedCases });
}
```

`ChatSvc.getRelatedCases` → new `ChatRepo.findLatestAssistantMessage(conversationId)`
(`orderBy: { createdAt: "desc" }`, `where: { role: "assistant" }`,
`include: { relatedCases: true }`) → return `.relatedCases?.items ?? []`.

**New ADR**: `ilovelawyer-api/docs/adr/0001-related-cases-tab.md` (first ADR in this
repo) — records consuming the `[RELATED_CASES]` frame, the persistence model, and the
conversation-scoped endpoint choice over threading a message id through the stream.

---

## 3. ilovelawyer-app — render the tab

**Install Tabs primitive** (none exists yet — `apps/web/components/ui/` currently only
has `custom-select.tsx`): `npx shadcn add tabs` → `apps/web/components/ui/tabs.tsx`.

**`apps/web/lib/chat/mutations.ts`** — add a query for the new endpoint and the shared
type:
```ts
export interface RelatedCase {
  type: string;
  title: string | null;
  url: string | null;
  caseNumber: string | null;
  raNumber: string | null;
  year: unknown;
  snippet: string | null;
  relevance: number | null;
  vetted: boolean;
}

export async function fetchRelatedCases(conversationId: string): Promise<RelatedCase[]> {
  const res = await apiFetchRaw(`/api/chat/conversations/${conversationId}/related-cases`);
  const { relatedCases } = await res.json();
  return relatedCases;
}
```

**`apps/web/app/(protected)/homepage/page.tsx`**:
- Extend `DisplayMessage` (lines 21-24) with `relatedCases?: RelatedCase[]`.
- After `sendChatMessage()`'s stream resolves (existing call around lines 234-249),
  call `fetchRelatedCases(conversationId)` and attach the result onto the just-completed
  assistant message in local state.

**`apps/web/components/chat/assistant-message.tsx`**:
- Add `relatedCases?: RelatedCase[]` prop.
- Wrap existing markdown output in `<Tabs>`: "Answer" tab = current `ReactMarkdown`
  render; "Related Cases" tab = a card list (title, case/RA number, year, snippet,
  link to `url`), only rendered when `relatedCases?.length` is truthy — otherwise
  render the answer alone with no tab strip, so non-legal replies are unaffected.

No ADR needed here — this repo has no `docs/adr/` convention (confirmed no matches
during exploration), so the design record for this leg lives in the ilovelawyer-api
ADR's "Consequences" section, cross-referenced.

---

## Verification

1. **chat-wonder-v2-api**: unit test `select_related_cases` with a synthetic pool
   containing a duplicate id across a search row and a `get_case` entry — assert the
   vetted entry wins and output is capped/sorted. Manually exercise `/chat-stream` (via
   `tracer.html` or a raw WS client) with a legal query and confirm a
   `[RELATED_CASES]{...}` frame appears before `__END__`, deduped and ranked.
2. **ilovelawyer-api**: send a legal-mode chat message end-to-end in a local run, then
   `GET /api/chat/conversations/:id/related-cases` and assert a non-empty, deduped list
   matching the schema above. Confirm a cache-hit replay (same question twice) still
   returns related cases.
3. **ilovelawyer-app**: run the app locally, ask a Philippine-law question, confirm the
   "Related Cases" tab appears with populated cards and working juris.ph links; ask a
   non-legal question and confirm no tab strip appears (empty `relatedCases`).
