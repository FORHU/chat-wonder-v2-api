#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Stage 1 of CW-6's required latency benchmark (chat-wonder-v2-api#147): how long the Decisions
API itself takes to answer a model_router.py-shaped question, in isolation. This is NOT a
turn-latency benchmark -- decide_turn_model/decide_tool_model are not raced against anything
here, and routing_overrides.py is never called, so neither the actual race nor an
ilovelawyer-api round trip confounds the number. What this measures is exactly what
MODEL_DECISION_TIMEOUT_MS needs to be set from: the Decisions API's own response-time
distribution for this call shape.

Stage 2 -- whole-turn latency with the race on vs off -- needs the minimal race wired into
the_server.py first (not built yet) and is a separate benchmark; see the module docstring
discussion in model_router.py and the plan's "Verification" section.

Requires live credentials: OPENAI_API_KEY (or whatever client.py's setup already expects) must
be set in the environment. This script makes real, billed calls to the Decisions API -- it is
not run in CI and not run by pytest. Run it against staging/prod-shaped traffic, never invented
numbers.

Usage:
    python benchmarks/benchmark_model_decision_latency.py --queries-file benchmarks/sample_queries.json --runs 3
    python benchmarks/benchmark_model_decision_latency.py --queries-file path/to/real_queries.json --runs 5 --out results.json

Each line of the queries file is one JSON object: {"persona": "...", "query": "...",
"default_model": "...", "candidates": ["...", "..."]}. Ship this with REAL questions sampled
from logs (the same way decisions_research.py's own PR #138/#139 benchmark did, "4 UK legal
questions x 3 runs per mode") -- the sample file checked in here is a placeholder shape, not
real traffic, and must not be mistaken for one.

Methodology note (learned from this repo's own precedent, PR #139): run every call
SEQUENTIALLY, never in parallel against the same services -- PR #139 flagged exactly that as a
confound in its own numbers ("All 6 runs also ran in parallel against the same services").
--delay-s adds a pause between calls if you also want to avoid rate limits, but the sequential
loop alone already avoids the contention problem.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from typing import List, Optional

# model_router.py lives at the repo root, one level up from this script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import dotenv  # noqa: E402

# the_server.py normally does this at import time, but model_router.py only imports from it
# lazily (inside _broadcast, to avoid a circular import) -- _best_candidate never touches that
# path, so nothing would load .env for this script otherwise.
dotenv.load_dotenv()

import model_router as router  # noqa: E402


@dataclass
class QuerySpec:
    persona: str
    query: str
    default_model: str
    candidates: List[str]
    # Free text, e.g. "real: local sessions.pkl, 2026-11-12" or "placeholder". Never blend a
    # "real" and a "placeholder" query's latencies into one number without saying so -- that's
    # exactly the kind of result that looks precise and isn't.
    source: str = "unspecified"


@dataclass
class CallResult:
    persona: str
    source: str
    latency_s: float
    chosen_model: Optional[str]
    confidence: Optional[float]
    error: Optional[str] = None


def load_queries(path: str) -> List[QuerySpec]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return [
        QuerySpec(
            persona=r["persona"],
            query=r["query"],
            default_model=r["default_model"],
            candidates=r["candidates"],
            source=r.get("source", "unspecified"),
        )
        for r in raw
    ]


def build_client():
    """A plain OpenAI client, deliberately not llm_usage.UsageTrackingOpenAI -- a Decisions API
    call is not a chat completion, and wrapping it would just pollute TurnUsage with nonsense."""
    from openai import OpenAI

    return OpenAI()


def run_once(client, spec: QuerySpec) -> CallResult:
    options = [m for m in spec.candidates if m != spec.default_model]
    t0 = time.perf_counter()
    try:
        best_model, best_p = router._best_candidate(client, spec.query, options)
        elapsed = time.perf_counter() - t0
        return CallResult(persona=spec.persona, source=spec.source, latency_s=elapsed, chosen_model=best_model, confidence=best_p)
    except Exception as e:  # noqa: BLE001 -- a benchmark run records the failure, it does not crash the run
        elapsed = time.perf_counter() - t0
        return CallResult(persona=spec.persona, source=spec.source, latency_s=elapsed, chosen_model=None, confidence=None, error=str(e))


def percentile(values: List[float], p: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    k = (len(ordered) - 1) * p
    f, c = int(k), min(int(k) + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def summarize(results: List[CallResult]) -> dict:
    ok = [r for r in results if r.error is None]
    failed = [r for r in results if r.error is not None]
    latencies = [r.latency_s for r in ok]
    return {
        "calls": len(results),
        "errors": len(failed),
        "error_rate": (len(failed) / len(results)) if results else 0.0,
        "p50_s": percentile(latencies, 0.50),
        "p95_s": percentile(latencies, 0.95),
        "p99_s": percentile(latencies, 0.99),
        "mean_s": statistics.fmean(latencies) if latencies else float("nan"),
        "max_s": max(latencies) if latencies else float("nan"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--queries-file", required=True, help="Path to a JSON file of query specs (see module docstring).")
    parser.add_argument("--runs", type=int, default=3, help="Repeats per query (default 3, matching this repo's PR #138/#139 precedent).")
    parser.add_argument("--delay-s", type=float, default=0.0, help="Pause between calls, in addition to running sequentially.")
    parser.add_argument("--out", default=None, help="Write raw per-call results + summary to this JSON file.")
    args = parser.parse_args()

    queries = load_queries(args.queries_file)
    if not queries:
        print("No queries loaded; nothing to benchmark.", file=sys.stderr)
        return 1

    try:
        client = build_client()
    except Exception as e:  # noqa: BLE001
        print(f"Could not build an OpenAI client: {e}\nSet OPENAI_API_KEY and try again -- this script needs live credentials.", file=sys.stderr)
        return 1
    results: List[CallResult] = []

    print(f"Running {len(queries)} query spec(s) x {args.runs} run(s), sequentially...")
    for spec in queries:
        for run_idx in range(args.runs):
            result = run_once(client, spec)
            results.append(result)
            status = f"error={result.error}" if result.error else f"chosen={result.chosen_model} p={result.confidence}"
            print(f"  [{spec.persona}/{spec.source}] run {run_idx + 1}/{args.runs}: {result.latency_s:.3f}s  {status}")
            if args.delay_s:
                time.sleep(args.delay_s)

    overall = summarize(results)
    by_persona = {}
    for persona in sorted({r.persona for r in results}):
        by_persona[persona] = summarize([r for r in results if r.persona == persona])
    by_source = {}
    for source in sorted({r.source for r in results}):
        by_source[source] = summarize([r for r in results if r.source == source])

    print("\n=== Summary (seconds) ===")
    print(f"Overall: {json.dumps(overall, indent=2)}")
    for persona, summary in by_persona.items():
        print(f"\n{persona}: {json.dumps(summary, indent=2)}")
    print("\n--- By source (never trust a number blending real and placeholder queries) ---")
    for source, summary in by_source.items():
        print(f"\n{source}: {json.dumps(summary, indent=2)}")

    if overall["errors"]:
        print(
            f"\n{overall['errors']} of {overall['calls']} calls errored -- decide_turn_model/decide_tool_model "
            "fail open on this in production, but a high error rate here means the benchmark numbers "
            "undercount how often the timeout path would actually matter.",
            file=sys.stderr,
        )

    if args.out:
        payload = {
            "overall": overall,
            "by_persona": by_persona,
            "by_source": by_source,
            "raw": [r.__dict__ for r in results],
        }
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote raw results and summary to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
