# Handoff: Thermo Chat Confidence Spans — chat-wonder-v2-api

**Last updated:** 2026-11-10 (verification session on a second machine)
**Branch:** `feat/thermo-chat`, pushed. No PR yet. **Squash-merge it**, because `de97e55` accidentally committed the whole `.venv` (about 3M lines). `40aa3d1` untracks it, but it stays in branch history.
**Status:** Core mechanism built, tested, and live-verified on two machines. Coverage and source-corpus bugs fixed on 2026-11-10; threshold calibration is the next step (see "Open issues").

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
| 4 testers (`model_tester_demo`, `legal_chat_demo`, `legal_uk_chat_demo`, `intake_chat_demo`) | "Confidence view" toggle (off by default; renamed from "Thermo view"), `[CONF]`/envelope guards, plus `[STRUCTURED_DATA]`/`[DONE]` guards where they were missing. |
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

### Fidelity to the Thermo Pulse paper: verified live 2026-11-10
Reference: Kim, M. H. (2026), "Thermo Pulse", Preprints.org, https://doi.org/10.20944/preprints202611.0277.v1 (CC BY 4.0). The citation is in `confidence.py` and in the uncertainty legend.

| Paper element | Here? | Evidence |
|---|---|---|
| H_t, renormalized top-5 entropy (Eq. 1) | Yes, gpt-4o*/4.1*/deepseek-chat | Matches an independent Eq. 1 to about 5e-12 on live gpt-4o-mini tokens. OpenAI `top_logprobs` are pre-temperature (unchanged between T=0 and T=1.8), which is what Eq. 1 needs. Mean H was 0.33 nats on a tight prompt, against the paper's 0.26–0.29. |
| Word display = max H over the word's tokens | Yes | Same rule as the reference UI; 0 alignment mismatches across 249 live words. |
| Purple threshold | Yes, 1.00 nats | The reference's default. The old 0.9×ln5 = 1.45 nats fired on 0% of tokens. |
| T_t, the PD temperature controller (Eq. 2), and the 5% cutoff | **No, impossible on OpenAI** | `temperature` is one number per request; a per-token array is rejected. |
| Any H_t on luna/terra (production legal) | **No** | Both APIs reject logprobs ("not supported with reasoning models"). This is why provenance spans exist. |

The reference's blue/orange/red text means the **temperature its controller applied**. We never set one, so those colours were removed rather than reused for entropy. The toggle is "Confidence view": it shows source provenance on legal answers and uncertainty on logprob-capable models, and the legend names which one.

## Open issues

### 1. Coverage and source corpus: FIXED 2026-11-10
These two bugs were found live and fixed together:
- **80-sentence cap.** `_candidate_sentences` scored only the first 80 sentences, which left 54 of 407 blocks shaded. It's now bounded only by `MAX_CANDIDATE_SENTENCES = 2000`, as a payload-safety bound.
- **Thin corpus.** `_confidence_sources()` used only `title + snippet`. It now flattens each pool row with `legal_citations.collect_tool_result_text_corpus([row])`, one IDF document per row, so full `legislation_get_section` text and PH `court_reasoning`/`full_text` reach the formula.
- **Smaller fixes in the same change:**
  - Ordered-list markers (`1. `) are stripped. Before, those spans could never match the DOM.
  - Quote characters are normalized the same way on both sides, so verbatim matches survive curly quotes.
  - Sources are normalized once per turn, not once per sentence.
  - The renderer caches block text, avoiding about 160K `textContent` reads on long answers.

**Live result** (same UK unfair-dismissal question):

| | Spans | QUOTED | SOURCED | UNSOURCED | Blocks shaded |
|---|---|---|---|---|---|
| Before | 71 | 0 | 4 | 67 | 54 / 407 |
| After | 757 | 12 | 490 | 255 | 658 / 658 |

The view renders in 44ms.

### 2. Formula constants are still starting guesses: next step
With the full corpus in place, calibration now measures the right thing. The visible weak spot is **short fragments**. Two-to-three-word list items ("witness statements", "appeal documents") come out SOURCED, because a full statute contains almost any common word pair. Consider a minimum distinctive-token count before SOURCED is allowed, or a length-aware threshold, then tune `SOURCED_COVERAGE_THRESHOLD`/`QUOTE_MIN_CHARS` on real traffic.

### 3. Sentences containing a citation link are unshaded apart from the link itself
This is by design (see `_candidate_sentences`' docstring), but it's still a visible gap on citation-dense paragraphs.

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
4. Click "Confidence view".

Expect green citation links and every sentence shaded, with markdown intact.

## Things worth remembering

- "Work on luna/terra" can't mean entropy: the Responses API exposes no logprobs. The provenance formula is the signal that works on every model.
- Self-consistency sampling worked but cost 4–5×, and it was an indirect proxy for provenance, which has a direct, free answer.
- Don't trust "fixed" without real output. Both open issues above were found only by driving the live tester, not by the 42 unit tests.
