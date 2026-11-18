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

`sample_queries.json` is a **placeholder shape**, not real traffic. Before this benchmark's
numbers are used to set `MODEL_DECISION_TIMEOUT_MS`, replace it with queries actually sampled
from logs, the same way `decisions_research.py`'s own PR #138/#139 benchmark did ("4 UK legal
questions × 3 runs per mode").

Run every call sequentially (the script already does this) — PR #139 flagged running in
parallel against the same services as a confound in its own numbers.
