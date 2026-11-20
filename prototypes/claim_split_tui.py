# -*- coding: utf-8 -*-
"""PROTOTYPE TUI -- throwaway shell over claim_split_prototype.py.

Run:  python prototypes/claim_split_tui.py   (from chat-wonder-v2-api/)

Lets you type a test sentence and a small source corpus, then shows, side by side:
  - the tier provenance_spans would assign to the WHOLE sentence today
  - the tier each atomic CLAIM UNIT gets after connective-splitting

The question: does splitting ever change the verdict in a way whole-sentence scoring
hides? Watch for cases where the whole-sentence tier is SOURCED (or UNSOURCED) while a
claim unit inside it is clearly QUOTED or clearly UNSOURCED on its own.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claim_split_prototype import build_idf, classify, normalize_sources, split_into_claims

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"

_TIER_COLOR = {"QUOTED": GREEN, "SOURCED": YELLOW, "UNSOURCED": RED}

DEFAULT_SOURCES = [
    "Section 98 of the Employment Rights Act 1996 provides that conduct amounting to "
    "gross misconduct justifies dismissal without notice.",
    "An employee with at least two years' continuous service may bring a claim for "
    "unfair dismissal to an employment tribunal.",
]

DEFAULT_SENTENCE = (
    "Many commentators criticise this approach as outdated, but section 98 of the "
    "Employment Rights Act 1996 provides that conduct amounting to gross misconduct "
    "justifies dismissal without notice."
)


def clear() -> None:
    print("\033[2J\033[H", end="")


def tier_tag(tier: str, score) -> str:
    color = _TIER_COLOR.get(tier, "")
    tag = f"{color}{BOLD}{tier}{RESET}"
    if score is not None:
        tag += f"  {DIM}C(S)={score:.2f}{RESET}"
    return tag


def render(state: dict) -> None:
    clear()
    print(f"{BOLD}Claim-unit split prototype{RESET}  {DIM}(confidence.py's real scoring, unmodified){RESET}\n")

    print(f"{BOLD}Sources ({len(state['sources'])}):{RESET}")
    for i, s in enumerate(state["sources"]):
        shown = s if len(s) <= 100 else s[:97] + "..."
        print(f"  {DIM}[{i}]{RESET} {shown}")
    print()

    print(f"{BOLD}Sentence:{RESET}")
    print(f"  {state['sentence']}\n")

    if state["result"] is None:
        print(f"{DIM}Press [c] to classify this sentence.{RESET}\n")
    else:
        whole_tier, whole_score, units = state["result"]
        print(f"{BOLD}Whole-sentence tier:{RESET}  {tier_tag(whole_tier, whole_score)}\n")

        if len(units) == 1:
            print(f"{DIM}No connective boundary found -- sentence is already one claim unit.{RESET}\n")
        else:
            print(f"{BOLD}Claim units after splitting ({len(units)}):{RESET}")
            for text, tier, score in units:
                print(f"  {tier_tag(tier, score)}")
                print(f"    {text}")
            print()

            tiers = {t for _, t, _ in units}
            if len(tiers) > 1 or (whole_tier not in tiers):
                print(f"{YELLOW}>> Splitting changed the picture: whole-sentence tier "
                      f"({whole_tier}) does not match every claim unit's tier.{RESET}\n")
            else:
                print(f"{DIM}Splitting agreed with the whole-sentence tier here.{RESET}\n")

    print(f"{BOLD}[c]{RESET}{DIM} classify{RESET}  "
          f"{BOLD}[s]{RESET}{DIM} new sentence{RESET}  "
          f"{BOLD}[o]{RESET}{DIM} add source{RESET}  "
          f"{BOLD}[r]{RESET}{DIM} reset sources{RESET}  "
          f"{BOLD}[q]{RESET}{DIM} quit{RESET}")


def do_classify(state: dict) -> None:
    norm_sources = normalize_sources(state["sources"])
    idf = build_idf(state["sources"])
    whole_tier, whole_score = classify(state["sentence"], norm_sources, idf)
    units_text = split_into_claims(state["sentence"])
    units = [(t, *classify(t, norm_sources, idf)) for t in units_text]
    units = [(t, tier, score) for (t, tier, score) in units]
    state["result"] = (whole_tier, whole_score, units)


def main() -> None:
    state = {
        "sources": list(DEFAULT_SOURCES),
        "sentence": DEFAULT_SENTENCE,
        "result": None,
    }
    do_classify(state)
    render(state)
    while True:
        cmd = input("\n> ").strip().lower()
        if cmd == "q":
            break
        elif cmd == "c":
            do_classify(state)
        elif cmd == "s":
            render(state)
            new_sentence = input("New test sentence: ").strip()
            if new_sentence:
                state["sentence"] = new_sentence
                state["result"] = None
        elif cmd == "o":
            render(state)
            new_source = input("New source text to add: ").strip()
            if new_source:
                state["sources"].append(new_source)
                state["result"] = None
        elif cmd == "r":
            state["sources"] = list(DEFAULT_SOURCES)
            state["result"] = None
        render(state)


if __name__ == "__main__":
    main()
