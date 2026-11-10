# Handoff: Thermo Chat Confidence Spans — chat-wonder-v2-api

**Last updated:** 2026-11-10 (verification session on a second machine)
**Branch:** `feat/thermo-chat`, pushed. No PR yet. **Squash-merge it**, because `de97e55` accidentally committed the whole `.venv` (about 3M lines). `40aa3d1` untracks it, but it stays in branch history.
**Status:** Core mechanism built, tested, and live-verified on two machines. One real coverage bug and one calibration gap are open; see "Open issues" first.

The old repo-root `thermo-chat-handoff.md` from earlier sessions is gone. This file and the architecture doc supersede it.

**Architecture doc (read this first):** https://claude.ai/artifact/UW9q2bRG4ybGktBQfevcrJ. It has the exact formula, the design history (citation/attribution reuse and then K-sample self-consistency were both discarded), the transport/safety diagrams, and the quote-anchoring mechanism.

---

## The one-paragraph version

Every sentence of a `legal`/`legal_uk`/`intake` answer gets one of three categories:

- **QUOTED**: near-verbatim from a source.
- **SOURCED**: the model's own conclusion, drawing on a source.
- **UNSOURCED**: no source at all.

A deterministic formula (IDF-weighted lexical overlap) assigns the category, not an LLM judge. It's wired into `the_server.py` and rendered client-side in all four dev testers. The renderer is a DOM overlay applied on top of `marked.parse()` output. The feature is gated behind `CONFIDENCE_SPANS` (default on). Don't put that flag in the committed `.env`: it's a behaviour flag, not a secret.

## Files on the branch

| File | What |
|---|---|
| `confidence.py` | `provenance_spans()` (the formula), `quoted_citation_spans()` (free href check), `token_entropy`/`group_words`/`entropy_payload` (per-token entropy, testers only), `build_payload`/`to_frame`. |
| `the_server.py` | `_confidence_sources()`, confidence assembly in the post-`__END__` extras phase, `[CONF]` live-frame guards in both websocket loops, `ChatRequest.confidence_live`, `ChatState.confidence_live`/`confidence_tokens` (stripped from pickle), `/version` `confidence` block. |
| `model_registry.py` | Came in with `2079417` (DeepSeek non-thinking routing: `get_client`, `shape_chat_args`). This work added `supports_logprobs()` (gpt-4o*, gpt-4.1*, deepseek-chat; not luna/terra/deepseek-reasoner). |
| `testers/confidence-render.js` | Shared renderer: `marked.parse()` first, unconditionally; spans overlaid per block element via `Range.extractContents()`; citation `<a>`s styled in place by href. |
| 4 testers (`model_tester_demo`, `legal_chat_demo`, `legal_uk_chat_demo`, `intake_chat_demo`) | "Thermo view" toggle (off by default), `[CONF]`/envelope guards, plus `[STRUCTURED_DATA]`/`[DONE]` guards where they were missing. |
| `tests/test_confidence.py` | 42 tests, including the regression test proving `[CONF]` frames can never leak into `full_response`. |

`_finalize_legal_response`, the citation pipeline and the doctrine guards are untouched.

## The formula

For each candidate sentence `S`, against this turn's retrieved sources `D`:

```
Q(S) = 1 if normalize(S) is a substring of some d ∈ D (or vice-versa, d ≥ 20 chars), else 0
idf(w) = ln(1 + N/df(w))
C(S) = Σ idf(w) for w ∈ T(S)∩vocab(D)  /  Σ idf(w) for w ∈ T(S)      (T = non-stopword words)

tier = QUOTED if Q=1;  SOURCED if C ≥ 0.45;  else UNSOURCED
```

A citation `<a href>` is QUOTED the moment its href resolves to a retrieved source. `SOURCED_COVERAGE_THRESHOLD = 0.45` and `QUOTE_MIN_CHARS = 20` are uncalibrated starting values.

## How the rendering works (read before touching confidence-render.js)

1. Markdown is always rendered first with `marked.parse()`, in every toggle state.
2. Spans are matched against block elements (`<p>`, `<li>`, `<h2>`, `<td>`…) using `Range.extractContents()`, not single-text-node search. That's because sentences cross `<strong>`/`<em>` boundaries.
3. Citation `<a>`s are matched by href (`span.ref`) and styled in place. They don't get a `.conf-span` wrapper, so DOM counts of `.conf-span` exclude them.
4. `confidence.py` extracts *cleaned* text that matches what `marked.parse()` renders, not raw markdown.

---

## Open issues

### 1. Coverage bug: only the first 80 sentences are ever scored
`confidence._candidate_sentences(text, limit=80)` stops at 80 sentences. Its own docstring says nothing should be left uncoloured, because an unshaded sentence reads as "never checked."

**Live evidence (2026-11-10):** a 31K-char UK unfair-dismissal answer had 407 text blocks, and only 54 got any shading. Everything after the first screenful is unshaded. The likely fix is to remove or raise the cap. The formula is cheap, and a cap should be about payload size rather than sentence count. After the fix, re-check payload size and render time on a long answer.

### 2. Most prose comes out UNSOURCED, because the source corpus is too thin
In that same answer, 67 of 71 spans were UNSOURCED and 4 were SOURCED. The 5 citation links were correctly QUOTED.

The cause is `_confidence_sources()`, which builds the corpus from search-result `title + snippet` only. The sources panel shows titles like "The right." and "General." The model actually read full statute text via `legislation_get_section`, and that text never reaches the formula. Fix the corpus before calibrating the threshold: include tool-returned section/judgment text. Tuning 0.45 against a corpus this thin would calibrate the wrong thing.

### 3. Formula constants are still starting guesses
Calibrate them after #1 and #2.

### 4. Legal Terminal surface: discovered, not integrated
`ilovelawyer-app/components/terminal/` (`law-panel.tsx` etc.) uses `/api/my-cases/:caseId/...` REST plus polled `AiJobStatus`, not `/chat-stream`. Start from `ilovelawyer-api/src/controllers/case-terminal.controller.ts` and `services/case-change-run.service.ts`.

### 5. Everything past chat-wonder-v2-api
Not started: relay event emission in `ilovelawyer-api`, Prisma `MessageConfidence` side table, app/client renderers. `ilovelawyer-api` was last seen on an unrelated branch (`test/deepseek-local-override`) with uncommitted work. Switch branches before touching it.

---

## Resolved this session

### UK test failure: pre-existing, not caused by this branch
`test_extent_not_assumed_uk_wide` failed 5/5 on the branch and 5/5 on `staging`. Two separate causes stacked:

1. **Stale local config (environment).** A gitignored `resources/functions/user_functions.env` held a "local cheap legal profile" with `LEGAL_USE_RESPONSES_API=false`, `LEGAL_CHAT_MODEL=gpt-4o-mini` and `LEGAL_REASONING_EFFORT=none`. It's loaded with `override=True`, so it beats `.env`.
   - The code now reads only `LEGAL_CHAT_MODEL_UK`/`_PH`, so the gpt-4o-mini override was silently ignored and the turn fell back to `gpt-5.6-luna` on Chat Completions.
   - OpenAI rejects that with a 400 ("Function tools with reasoning_effort are not supported for gpt-5.6-luna in /v1/chat/completions").
   - This broke **all 6** live legal regression tests and live legal chat on the dev server. Commenting out the 3 lines fixed it.
   - **Code smell worth a follow-up:** the documented `LEGAL_USE_RESPONSES_API=false` rollback path can no longer work with the default luna model.
2. **Brittle assertion (test).** On the real Responses path, the test went 1/3 on the branch and 1/3 on `staging`. All 6 answers were legally correct and flagged that NI has its own legislation. The fixed phrase list misses natural wording such as "Employment Rights **(Northern Ireland) Order** 1996", "own **employment** legislation" and "separate employment-law **framework**". Fix this in its own small PR by loosening the phrases into regexes.

The other known failures are unrelated to this feature:
- `test_responses_store.py::test_everything_else_is_passed_through_and_the_result_returned`: pre-existing on HEAD.
- `test_no_other_code_calls_the_responses_api_directly`: fails only on machines with an `.agent-office/worktrees/` copy inside the repo, which the scan picks up. Environmental.

### Dev server hang: did not reproduce
The hang (port 8000 accepting TCP but never responding) was on the original machine's long-running `--reload` process. On a fresh `uvicorn the_server:app --port 8000` without `--reload`, startup is clean and `/version` answers in about 3ms with the `confidence` block. A live UK turn completed in 84s. If it recurs, look at that process's console first.

### Branch hygiene
`de97e55` committed `.venv/` (11,852 files, pointing at another machine's Python 3.10, unusable elsewhere) and a stray `ssm-params.json`. Both are untracked and gitignored as of `40aa3d1`. Recreate the venv locally: `py -3.12 -m venv .venv && .venv/Scripts/python -m pip install -r requirements.txt pytest`.

---

## Verification commands

```bash
cd chat-wonder-v2-api
python -m pytest tests/test_confidence.py -q          # 42 pass
python -m pytest tests/ test_responses_store.py -q    # live legal tests need a working legal env (see above)
curl -s localhost:8000/version                         # check the `confidence` block
```

To see it work:
1. Start the server fresh.
2. Serve `testers/` over HTTP (`python -m http.server`), since Playwright blocks `file://`.
3. Open `legal_uk_chat_demo.html` and ask a UK question.
4. Click "Thermo view".

Expect green citation links and shaded sentences with markdown intact. Until open issue #1 is fixed, shading covers only the first ~80 sentences.

## Things worth remembering

- "Work on luna/terra" can't mean entropy: the Responses API exposes no logprobs. The provenance formula is the signal that works on every model.
- Self-consistency sampling worked but cost 4–5×, and it was an indirect proxy for provenance, which has a direct, free answer.
- Don't trust "fixed" without real output. Both open issues above were found only by driving the live tester, not by the 42 unit tests.
