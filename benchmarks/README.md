# Benchmarks

## `benchmark_model_decision_latency.py` (CW-6, chat-wonder-v2-api#147)

Measures the Decisions API's own response-time distribution for a `model_router.py`-shaped
question, in isolation — not a turn-latency benchmark, and not the racing integration itself
(neither exists to benchmark yet). This is stage 1 of the plan's required latency benchmark;
see the script's own module docstring for the full methodology and stage 2.

```
python benchmarks/benchmark_model_decision_latency.py --queries-file benchmarks/sample_queries.json --runs 3 --out results.json
```

Requires a real `OPENAI_API_KEY` in the environment — it makes real, billed calls. Not run in CI
or by pytest.

`sample_queries.json` is a **hand-written, source-grounded** query set covering the three
personas CW-6/CW-7 actually route for (`legal_uk`, `legal`, `intake` — `stylist` and the other
consumer-facing personas run on `CHAT_MODEL`/gpt-4o-mini, not the luna/terra candidates this
routing targets, so they're out of scope). Each entry's `query` matches that persona's real tool
whitelist and system prompt in `process_persona` (the_server.py) / `intake_prompt.txt`, not an
invented shape.

This isn't log-sampled the way `decisions_research.py`'s PR #138/#139 benchmark was, because
there's nothing to sample from: `the_server.py` never logs real query text anywhere persistent
(confirmed by reading it end to end — `broadcast_trace`'s SSE event is in-memory only, no
`logging.*` call prints `user_input`), and the one candidate source, `sessions.pkl`, is wiped on
every deploy and was, on inspection, >99% internal summariser-prompt text rather than user
queries, with zero representative traffic on staging and prod UK's box stopped. One entry here
(`"source": "real: local sessions.pkl, 2026-11-12"`) is a genuine extracted query kept for
comparison; every other entry is `"source": "handwritten: ..."` and says exactly what source-code
section it's grounded in. Never blend `real` and `handwritten` latencies into one number without
checking `by_source` in the benchmark's own output — see the script's module docstring.

Run every call sequentially (the script already does this) — PR #139 flagged running in
parallel against the same services as a confound in its own numbers.
