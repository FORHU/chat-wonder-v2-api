#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extracts real queries for benchmark_model_decision_latency.py's queries-file, from a running
chat-wonder-v2-api instance's own session snapshot (sessions.pkl, SESSION_STORAGE_PATH in
the_server.py) -- the ONLY place a real user_input ever survives. the_server.py logs no real
query text anywhere else: broadcast_trace's "New turn" event is an in-memory SSE broadcast, never
written to a log file or stdout, and no logging.info call anywhere prints user_input. There is no
CloudWatch (or any other) log export command for this, because there is nothing there to export.

Privacy: this reads ONLY the session's `current_query` field. It deliberately never touches or
emits case_id, user_id, case_document_manifest, or anything else on the session -- those stay on
the box. Run this ON the instance (or against a copy of sessions.pkl you already control), and
hand the resulting small JSON file to whoever needs it, not the raw pickle.

Usage (on the EC2 instance, inside the running container):
    docker exec chat-wonder-v2 python benchmarks/extract_real_queries.py --out /tmp/real_queries.json
    docker cp chat-wonder-v2:/tmp/real_queries.json ./real_queries.json

Or via the same SSM mechanism the deploy workflow already uses -- see the module's "SSM one-
liner" section below.

What gets skipped:
  - An empty current_query (most saved sessions have one -- a session only carries a live query
    for the brief window between a turn starting and the state being updated for the next one).
  - The internal title-generation prompt ("Create a concise title for a ... consultation...") --
    that is a different feature's own prompt template, not a real user turn.
  - Anything shorter than --min-chars (default 40): too short to be a representative turn query.

Output shape: a list of {"persona", "query", "source"}. `default_model`/`candidates` are NOT
filled in -- this script has no way to know which models are actually configured for a persona
on this deployment (LEGAL_CHAT_MODEL_UK etc. are env-specific). Add those fields by hand (or
merge with sample_queries.json's existing entries for that persona) before passing the result to
benchmark_model_decision_latency.py.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from typing import List, Optional

# Unpickling a ChatState needs the_server importable (pickle resolves classes by module path),
# and the_server.py lives at the repo root, one level up from this script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_TITLE_PROMPT_PREFIX = "create a concise title for a"

_PERSONA_TAGS = [
    ("[legal ai uk]", "legal_uk"),
    ("[legal ai]", "legal"),
    ("[stylist]", "stylist"),
    ("[intake]", "intake"),
]


def detect_persona(query: str) -> str:
    lowered = query.strip().lower()
    for tag, persona in _PERSONA_TAGS:
        if lowered.startswith(tag):
            return persona
    return "unknown"


def strip_tag(query: str) -> str:
    return re.sub(r"^\s*\[[^\]]+\]\s*", "", query).strip()


def extract(path: str, min_chars: int) -> List[dict]:
    with open(path, "rb") as f:
        sessions = pickle.load(f)

    out = []
    for state in sessions.values():
        query: Optional[str] = getattr(state, "current_query", None)
        if not query or not query.strip():
            continue
        if query.strip().lower().startswith(_TITLE_PROMPT_PREFIX):
            continue
        if len(query.strip()) < min_chars:
            continue
        out.append({
            "persona": detect_persona(query),
            "query": strip_tag(query),
            "source": "real: extracted sessions.pkl",
        })
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sessions-file", default="sessions.pkl", help="Path to the session snapshot (default: sessions.pkl, the_server.py's own default path).")
    parser.add_argument("--min-chars", type=int, default=40, help="Drop queries shorter than this (default 40).")
    parser.add_argument("--out", default=None, help="Write the JSON array here; prints to stdout if omitted.")
    args = parser.parse_args()

    try:
        rows = extract(args.sessions_file, args.min_chars)
    except FileNotFoundError:
        print(f"No session file at {args.sessions_file}.", file=sys.stderr)
        return 1

    payload = json.dumps(rows, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(payload)
        print(f"Wrote {len(rows)} real quer{'y' if len(rows) == 1 else 'ies'} to {args.out}", file=sys.stderr)
    else:
        print(payload)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# ─────────────────────────────────────────────────────────────────────────────
# SSM one-liner (same mechanism .github/workflows/deploy.yml already uses to run commands on
# the box -- adapt EC2_TARGET_TAG_KEY/VALUE to match that workflow's own values):
#
#   aws ssm send-command \
#     --document-name "AWS-RunShellScript" \
#     --targets "Key=tag:<EC2_TARGET_TAG_KEY>,Values=<EC2_TARGET_TAG_VALUE>" \
#     --parameters commands='[
#       "docker exec chat-wonder-v2 python benchmarks/extract_real_queries.py --out /tmp/real_queries.json",
#       "docker cp chat-wonder-v2:/tmp/real_queries.json /tmp/real_queries.json",
#       "cat /tmp/real_queries.json"
#     ]' \
#     --query "Command.CommandId" --output text
#
# Then, once that command finishes:
#
#   aws ssm get-command-invocation --command-id <id-from-above> --instance-id <instance-id> \
#     --query "StandardOutputContent" --output text > real_queries.json
#
# The file is small (a handful of short JSON objects) and carries no PII fields, so it fits
# comfortably in SSM's command-output capture without needing S3.
# ─────────────────────────────────────────────────────────────────────────────
