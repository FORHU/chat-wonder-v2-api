# Confidence Spans — Implementation Status (Session 3)

**Date:** 2026-11-10
**Branch:** `feat/thermo-chat`
**Mode:** Autonomous ("auto mode"), Sonnet 5, unsupervised. **Review this before testing.**

This supersedes `thermo-chat-handoff.md` (the session-1/session-2 handoff, which documented a
failed attempt and false "COMPLETED" claims — see its own "Mistakes made" section). That document
can be deleted once this one is reviewed; nothing in it reflects the current code.

Full architecture, rationale, and diagrams: the published artifact from the planning session —
ask the user for the link, or see `C:\Users\bulan\.claude\plans\c-users-bulan-onedrive-desktop-thermocha-cozy-plum.md`
for the text plan it was built from.

---

## What's actually true right now (verify before trusting this doc)

```
git log --oneline -1          # should NOT be committed yet — nothing here has been committed
git status                    # see the file list below
python -m pytest tests/test_confidence.py -v   # 39 tests, all passing at time of writing
python -m pytest tests/ test_responses_store.py -q   # 614 passed, 1 pre-existing failure (see below)
```

**Nothing has been committed.** All changes are in the working tree, unstaged/staged exactly as
`git status` shows. This is deliberate — the user asked for this to be reviewable before testing,
and a commit should be the user's own decision after that review.

**One pre-existing test failure, unrelated to this work:**
`test_responses_store.py::test_everything_else_is_passed_through_and_the_result_returned` fails on
a clean `feat/thermo-chat` checkout too (verified by `git stash` and re-running it in isolation).
Do not spend time on it as part of reviewing this feature.

---

## What was implemented (Phases 0–2, chat-wonder-v2-api only)

### New files
- **`confidence.py`** — the signal-agnostic span layer. Entropy math (renormalized top-5 Shannon
  entropy, `token_entropy`/`group_words`/`entropy_payload`), live-frame encoding (`to_live_frame`),
  verification spans (`attribution_spans`, `citation_spans`), and payload assembly
  (`build_payload`, `to_frame`). Read its module docstring first — it explains the two
  divergences from the Thermo Chat reference (renormalization; no fabricated "temperature") and
  why GROUNDED requires a verified quote rather than trusting the judge's `supported` verdict.
- **`testers/confidence-render.js`** — shared client-side renderer for all four testers. One file,
  not four copies, specifically because duplicated threshold constants are how the Thermo Chat
  reference's own slider desynced from itself.
- **`tests/test_confidence.py`** — 39 tests. The one that matters most is
  `ChatWonderStreamingTests.test_conf_frames_never_reach_full_response`: it drives the real
  `streaming_run_function_chain` (not a mock of it) with fake logprobs and asserts the exact prior
  failure mode (interleaved metric frames swallowing the whole buffered answer) cannot recur.

### Modified files
- **`model_registry.py`** — `supports_logprobs()` (allowlist: `gpt-4o*`, `gpt-4.1*`,
  `deepseek-chat`; explicitly NOT `gpt-5.6-luna`/`terra`/`deepseek-reasoner`) and
  `logprob_capable_prefixes()`. `shape_chat_args()` now strips `logprobs`/`top_logprobs` for any
  model outside that allowlist, defensively, even though `perform_chat` already gates the request.
- **`the_server.py`**:
  - `ChatRequest.confidence_live: Optional[bool] = False` — the dev-tester opt-in for live
    `[CONF]` frames. The relay never sends this field; production cannot receive live frames even
    if a server-side flag is flipped by mistake.
  - `ChatState.confidence_live` / `ChatState.confidence_tokens` — both stripped from the pickle in
    `__getstate__` (per-turn telemetry, no reason to survive a restart).
  - `streaming_run_function_chain`: computes `_telemetry_on` / `_confidence_live` once per turn,
    requests `logprobs`/`top_logprobs=5` in `perform_chat` when eligible, captures entropy from
    `chunk.choices[0].logprobs.content[]` (never from the already-transformed `part` text) into
    `state.confidence_tokens`, and yields `[CONF]{...}[/CONF]` live frames when `confidence_live`
    is set.
  - Both websocket loops: a `[CONF]` guard placed immediately beside the existing `[TRACE]` guard
    (`send_text` then `continue`) — this `continue` is the entire safety property; see the
    regression test above for what happens without it.
  - Post-`__END__` extras phase: builds and sends the `{"type":"confidence",...}` envelope for
    `legal`/`legal_uk`/`intake` when `full_response` was non-empty, wrapped in its own
    try/except so a telemetry failure can never look like a failed turn to ilovelawyer-api (same
    reasoning the existing extras code already uses for reasoning/decisions/mind-map).
  - `/version` now advertises a `confidence` capability block.
- **`testers/model_tester_demo.html`, `legal_chat_demo.html`, `legal_uk_chat_demo.html`,
  `intake_chat_demo.html`** — each gets: the shared renderer script tag, a "Thermo view" toggle in
  the header (off by default — toggle off renders byte-identically to before this change), a
  legend container, `[CONF]`-frame and `{"type":"confidence"}`-envelope guards placed before the
  text-accumulation fallthrough, and a `lastBubble`/`lastRaw` pair so the envelope (which arrives
  **after** `__END__`, once `finishAssistantTurn()` has already cleared the streaming variables)
  can still annotate the just-finished bubble.

### Incidental fixes made while wiring the above (not the point of this session, but low-risk and directly adjacent to what was being edited)
- **`legal_chat_demo.html` and `legal_uk_chat_demo.html` had no `[TRACE]` branch at all** — a
  trace frame fell straight into the answer bubble and corrupted it. This was a live bug
  independent of confidence spans; fixed as part of adding the guards in the same spot.
  `intake_chat_demo.html` had the same gap (no `[TRACE]` branch), also fixed.
- **`legal_chat_demo.html` never dropped `[RELATED_CASES]`** (every other tester does). Fixed in
  the same edit as the `[Sources]` handling.

---

## What was deliberately NOT done, and why

### Phase 0.1 (relay: ignore unknown envelopes in ilovelawyer-api) — NOT applied
The plan calls for a one-line safety fix in `ilovelawyer-api/src/utils/chatWonder.ts` (and
`intakeChatWonder.ts`): add a final `else { return; }` to the envelope-type branch so an
unrecognized `{"type":...}` frame is dropped instead of falling through into answer prose.

**This was not applied.** `ilovelawyer-api`'s current checkout is on `test/deepseek-local-override`
with the user's own uncommitted changes already in its working tree (`src/config.ts`, `yarn.lock`).
Editing there would have mixed an unrelated change into someone else's in-progress branch state.
The exact diff to apply, once the user is on the right branch for it:

```ts
// In chatWonder.ts and intakeChatWonder.ts, wherever the envelope-type switch/if-chain lives
// (reasoning / decisions / usage cases): add a final default that drops anything unrecognized.
} else {
  return; // unknown envelope type — never let it fall through to answer text
}
```

**Practical effect of not having this yet:** chat-wonder-v2-api now emits a
`{"type":"confidence",...}` envelope for legal/legal_uk/intake turns (whenever `CONFIDENCE_SPANS`
is enabled, which defaults to true). ilovelawyer-api's relay does not yet know what to do with an
envelope of type `"confidence"` and — per the architecture doc's finding — **currently has no
default-ignore case**, so this frame could fall through into answer prose on the relay side.

**This means: do not point a running ilovelawyer-api instance at a chat-wonder-v2-api deployment
of this branch until either (a) Phase 0.1 is applied to the relay, or (b) `CONFIDENCE_SPANS=false`
is set in chat-wonder-v2-api's environment.** The dev testers are safe to use as-is (they talk to
chat-wonder-v2-api directly, with their own guards now in place) — only the relay path is at risk.

### Phases 3–7 — not started
Relay event emission (`chat:confidence`), the Prisma `MessageConfidence` migration, the app
renderer/toggle, the client app wiring, and the intake verifier are all specified in detail in the
architecture plan but require work in `ilovelawyer-api`, `ilovelawyer-app`, and
`ilovelawyer-client` — three repos this session did not touch, partly for the branch-state reason
above and partly because schema migrations and production frontend changes deserve a human in the
loop before they're written, not after.

### Risk controls (quote verification, pass-consensus) — implemented, not yet calibrated
`attribution_spans` in `confidence.py` already implements both risk controls from the plan:
- `GROUNDED` requires the judge's quote to verify (normalized substring match) against the
  exhibit text; a `supported` verdict with no verifying quote becomes `INFERENCE`, never `GROUNDED`.
- `UNSUPPORTED` requires **every** judge pass to agree; a claim the passes disagreed about is
  omitted entirely rather than shown.

**What's NOT done: the calibration gate itself.** The plan's pre-registered numbers (≥90%
precision on human review, marks on ≤10% of in-scope sentences) have not been measured against
real traffic, because nothing has run against real traffic yet. **Do not expose this to clients
(Phase 5/6) before that measurement happens.** The entropy display thresholds in
`confidence-render.js` are explicitly marked as starting guesses for the same reason — see the
`ENTROPY_BANDS` comment in that file.

---

## How to verify this yourself before testing

1. **Read `confidence.py`'s module docstring and `attribution_spans`' docstring first.** They
   explain the two most consequential design choices (quote verification, pass-consensus) and
   why each exists.
2. **Run the regression test by itself** and read what it asserts:
   ```
   python -m pytest tests/test_confidence.py::ChatWonderStreamingTests -v
   ```
3. **Diff `the_server.py`** (`git diff the_server.py`) and confirm every `[CONF]` guard is
   immediately followed by `continue`, exactly like the adjacent `[TRACE]` guard. This is the one
   property that, if missing, reproduces the documented prior failure.
4. **Start the server locally and open `testers/model_tester_demo.html`.** Select `gpt-4o-mini` or
   `deepseek-chat` as the model override (the only two with real entropy support), send a message
   with persona "General", and toggle "Thermo view" on — live per-word coloring should appear
   during streaming. With `gpt-5.6-luna` selected, the toggle should show "confidence shading
   unavailable" rather than coloring anything (no logprobs on that model).
5. **Open `legal_chat_demo.html` or `legal_uk_chat_demo.html`, ask a legal question that triggers
   search**, and after the answer finishes, toggle "Thermo view" on. You should see sentence-level
   shading appear, sourced from the citation/attribution guards. If a sentence you know is fine
   shows as UNSUPPORTED, that is exactly the false-positive risk the calibration gate exists to
   catch — note it, don't ship past it.
6. **Confirm `CONFIDENCE_SPANS=false` disables everything**: no logprobs requested (check the
   outbound args in a debugger or log), no envelope sent, all four testers render exactly as they
   did before this branch.
7. **Do not set `CONFIDENCE_SPANS` or any other behavior flag in the committed `.env`.** It's a
   behavior flag, not a secret, and this repo's `.env` is committed — see the memory rule about not
   copying behavior flags into live env.

---

## Open questions for the user (none of these blocked the work above, but they matter for Phases 3+)

- **Calibration data collection**: who runs the representative-query sweep the plan's
  "Calibration" section calls for, and against which persona/jurisdiction first?
- **Intake's `UNSUPPORTED` tier**: the plan says intake launches with `GROUNDED`/`INFERENCE` only,
  no `UNSUPPORTED`, until its own precision data exists. The Phase 7 verifier isn't built yet, so
  this hasn't come up in practice — flagging it now so it isn't missed when Phase 7 starts.
  `confidence.py`'s `checked=False` path for intake already reflects "nothing built yet" (coverage
  reads `"none"`), not a decision about the tier itself.
- **Which repo gets Phase 0.1 first**: `ilovelawyer-api` needs a clean branch state before that
  one-line relay fix can land without mixing into unrelated work.
