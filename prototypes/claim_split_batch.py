# -*- coding: utf-8 -*-
"""PROTOTYPE -- throwaway batch driver over claim_split_prototype.py.

Run:  python prototypes/claim_split_batch.py   (from chat-wonder-v2-api/)

The interactive TUI is good for driving one sentence at a time by hand. This runs a
bank of realistic (and deliberately adversarial) UK-legal-style sentences through the
same split+classify logic in one shot, so the split function can be stress-tested against
more cases than one person will patiently type into a prompt. Each case states what it's
checking for; the summary at the bottom says which cases needed splitting to be correct,
which didn't, and whether the splitter ever misfired (no boundary where one was expected,
or a boundary where none should exist).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, NamedTuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from prototypes.claim_split_prototype import (
    build_idf,
    classify,
    normalize_sources,
    split_into_claims,
    split_into_claims_guarded,
)

BOLD = "\x1b[1m"
DIM = "\x1b[2m"
RESET = "\x1b[0m"
GREEN = "\x1b[32m"
YELLOW = "\x1b[33m"
RED = "\x1b[31m"
CYAN = "\x1b[36m"

_TIER_COLOR = {"QUOTED": GREEN, "SOURCED": YELLOW, "UNSOURCED": RED}


def _tag(tier: str) -> str:
    return f"{_TIER_COLOR.get(tier, '')}{tier}{RESET}"


class Case(NamedTuple):
    name: str
    sentence: str
    sources: List[str]
    expect_split: bool  # does this sentence contain a genuine clause boundary?
    note: str
    adversarial: bool = False  # True = deliberately probing for a guard failure, not a realistic case


ERA_1996_98 = (
    "Section 98 of the Employment Rights Act 1996 provides that conduct amounting to "
    "gross misconduct justifies dismissal without notice."
)
TRIBUNAL_SERVICE = (
    "An employee with at least two years' continuous service may bring a claim for "
    "unfair dismissal to an employment tribunal."
)
REASONABLE_RESPONSE = (
    "The tribunal must consider whether dismissal fell within the band of reasonable "
    "responses open to a reasonable employer in the circumstances."
)

CASES: List[Case] = [
    Case(
        name="editorial opinion masked by trailing quote (original finding)",
        sentence=(
            "Many commentators criticise this approach as outdated, but section 98 of "
            "the Employment Rights Act 1996 provides that conduct amounting to gross "
            "misconduct justifies dismissal without notice."
        ),
        sources=[ERA_1996_98, TRIBUNAL_SERVICE],
        expect_split=True,
        note="UNSOURCED opinion + QUOTED statute -- whole-sentence scoring should hide the opinion.",
    ),
    Case(
        name="reversed order: quote first, opinion second",
        sentence=(
            "Section 98 of the Employment Rights Act 1996 provides that conduct "
            "amounting to gross misconduct justifies dismissal without notice, "
            "although many practitioners consider the threshold too permissive."
        ),
        sources=[ERA_1996_98, TRIBUNAL_SERVICE],
        expect_split=True,
        note="Same masking risk in the other direction -- QUOTED clause first, UNSOURCED opinion trailing.",
    ),
    Case(
        name="three clauses, two connectives",
        sentence=(
            "The tribunal found in the claimant's favour; however, the respondent's "
            "conduct, although serious, did not amount to gross misconduct justifying "
            "summary dismissal."
        ),
        sources=[ERA_1996_98],
        expect_split=True,
        note="Nested connective inside a clause ('although serious') -- checks the splitter handles >2 units.",
    ),
    Case(
        name="simple sentence, no connective",
        sentence="The claimant was dismissed on 3 March 2026 without prior warning.",
        sources=[ERA_1996_98, TRIBUNAL_SERVICE],
        expect_split=False,
        note="Negative control -- most real sentences. Splitter must leave this alone.",
    ),
    Case(
        name="'but' with no preceding comma (not a clause boundary)",
        sentence="There was nothing but speculation behind the respondent's account of events.",
        sources=[ERA_1996_98],
        expect_split=False,
        note="Regex requires a preceding comma/semicolon -- this 'but' is inside a noun phrase, must NOT split.",
    ),
    Case(
        name="'however' as a bare intensifier, no preceding comma",
        sentence="However unreasonable the respondent's conduct appeared, the tribunal applied the correct test.",
        sources=[REASONABLE_RESPONSE],
        expect_split=False,
        note="Sentence-initial 'However' with no preceding comma -- must NOT be treated as a boundary.",
    ),
    Case(
        name="'provided that' as a genuine subordinate clause",
        sentence=(
            "The employer may rely on the band of reasonable responses test, provided "
            "that the dismissal decision fell within that band given all the circumstances."
        ),
        sources=[REASONABLE_RESPONSE],
        expect_split=True,
        note="Multi-word connective; checks the longest-match-first ordering in _CONNECTIVES.",
    ),
    Case(
        name="elided quotation combined with a connective (corrected)",
        # Every fragment either side of the ellipsis is a VERBATIM substring of the
        # source -- the first attempt at this case failed because the wrapper text
        # ("The tribunal cited the test that...") wasn't itself drawn from the source,
        # so the elided-quotation path (requires ALL fragments to match) correctly
        # rejected it. This version elides only WITHIN a genuine quotation.
        sentence=(
            "The tribunal must consider whether dismissal fell … open to a "
            "reasonable employer in the circumstances, but applied the test narrowly "
            "on the facts of this case."
        ),
        sources=[REASONABLE_RESPONSE],
        expect_split=True,
        note="First unit should resolve QUOTED via the elided-quotation path (both fragments are real source substrings) after splitting at ', but'.",
    ),
    Case(
        name="no sources available at all",
        sentence="The claimant's conduct was reasonable, but the dismissal was procedurally unfair.",
        sources=[],
        expect_split=True,
        note="Empty corpus -- both whole and split must come back UNSOURCED, not crash.",
    ),
    Case(
        name="very short clause after split",
        sentence="The claim succeeded, but barely.",
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="Second unit is two words -- checks the splitter doesn't produce a junk empty/trivial fragment.",
    ),
]


SECTION_PROVISO = (
    "This section applies to every contract of employment, but shall not apply where "
    "the employee was fairly dismissed for gross misconduct."
)

# Deliberate attempts to break the comma-gating guard (requires a preceding comma/
# semicolon before the connective). The guard protects against a BARE connective with
# no preceding punctuation; these go after cases where a comma IS present but the
# connective still isn't marking a genuine two-independent-clause boundary.
ADVERSARIAL_CASES: List[Case] = [
    Case(
        name="[ADV] concessive parenthetical, comma on both sides",
        sentence=(
            "The respondent's conduct, although serious, did not amount to gross "
            "misconduct justifying summary dismissal."
        ),
        sources=[ERA_1996_98],
        expect_split=True,  # the guard WILL fire -- the point is to see what it produces
        note="',although X,' is a parenthetical insertion, not a clause boundary -- "
             "check whether the resulting units are still complete, standalone claims.",
    ),
    Case(
        name="[ADV] contrastive parenthetical, comma on both sides",
        sentence=(
            "The tribunal, while sympathetic to the claimant's circumstances, could not "
            "depart from the statutory band of reasonable responses test."
        ),
        sources=[REASONABLE_RESPONSE],
        expect_split=True,
        note="Same pattern with 'while' -- subject and verb get torn apart if the "
             "splitter treats the opening comma as a clause boundary.",
    ),
    Case(
        name="[ADV] idiomatic adverbial, comma on both sides",
        sentence="The claim was, yet again, dismissed without a substantive hearing.",
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="'yet again' is a fixed adverbial phrase, not a coordinating conjunction -- "
             "splitting here would separate 'was' from its complement 'dismissed'.",
    ),
    Case(
        name="[ADV] Oxford-comma list coordination",
        sentence="The tribunal reviewed the contract, the handbook, but not the appendix.",
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="'but' here coordinates list ITEMS, not independent clauses -- splitting "
             "produces a fragment ('but not the appendix.') with no verb of its own.",
    ),
    Case(
        name="[ADV] connective inside an otherwise-verbatim quotation",
        sentence=SECTION_PROVISO,
        sources=[SECTION_PROVISO],
        expect_split=True,
        note="Sentence is a word-for-word copy of the source, which itself contains "
             "', but shall not apply...'. Checks that splitting a genuine verbatim "
             "quotation in half doesn't break QUOTED on either resulting half.",
    ),
]

# ROUND 2 -- targeting the guard's own weak spot: _opens_verbless_aside treats any word
# ending "-ed"/"-ing" as proof of a real verb, which also matches a bare PAST PARTICIPLE
# used adjectivally in a reduced clause with no subject of its own ("although delayed,").
# That's one of the single most common reduced-clause patterns in formal/legal English,
# so if the guard is fooled by it, this is exactly the kind of case worth finding now.
ADVERSARIAL_ROUND2: List[Case] = [
    Case(
        name="[ADV2] reduced past-participle aside (predicted guard failure)",
        sentence=(
            "The claimant's appeal, although delayed, was accepted by the tribunal."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,  # the (flawed) guard is expected to still fire here
        note="'delayed' has no subject of its own -- it's a reduced aside ('[being] "
             "delayed'), not a clause. Predicted: the '-ed' check wrongly treats it as a "
             "verb and lets the split through, tearing 'The claimant's appeal' from its "
             "own predicate 'was accepted'.",
    ),
    Case(
        name="[ADV2] reduced past-participle aside, second connective",
        sentence=(
            "The employer, although advised against it, proceeded with the dismissal."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="Same pattern with 'advised' -- same predicted failure mode.",
    ),
    Case(
        name="[ADV2] genuine clause, internal list comma (must still split)",
        sentence=(
            "The claimant succeeded on the first ground, while the tribunal rejected "
            "the second, third, and fourth grounds advanced."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="'while' here opens a GENUINE clause with its own subject ('the tribunal') "
             "and verb ('rejected') -- the comma after 'rejected the second' is List "
             "punctuation, not an aside boundary. Any fix for the round-1 cases must not "
             "start guarding this one by mistake.",
    ),
    Case(
        name="[ADV2] two reduced asides, no top-level clause boundary at all",
        sentence=(
            "The tribunal held, although reluctantly, that the claimant's appeal, while "
            "procedurally irregular, succeeded on the merits."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=False,  # nothing here is a genuine independent-clause boundary
        note="Both 'although reluctantly' and 'while procedurally irregular' are reduced "
             "asides modifying a single clause ('The tribunal held that the claimant's "
             "appeal succeeded on the merits'). Correct behaviour: the sentence stays ONE "
             "unit -- no connective here should be treated as a split point.",
    ),
]

# ROUND 3 -- the subject heuristic's own predicted blind spot: it recognises a subject by
# a CLOSED list of determiners/pronouns plus capitalisation. A genuine subordinate clause
# whose subject is a bare common noun with NO article ("government policy shifted...",
# not "the government's policy shifted...") has the same lowercase, non-pronoun surface
# shape as a reduced aside -- predicted to be a miss, not a false tier. Also fills in
# 'whereas'/'though', untested so far.
ADVERSARIAL_ROUND3: List[Case] = [
    Case(
        name="[ADV3] genuine clause, bare-noun subject with no article (predicted miss)",
        sentence=(
            "The claimant withdrew the application, although government policy shifted "
            "considerably during the appeal process."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="'government policy' is a real subject (no article), but the heuristic only "
             "recognises determiners/pronouns/capitalised words -- predicted: wrongly "
             "treated as a subjectless aside and the genuine boundary is missed.",
    ),
    Case(
        name="[ADV3] 'whereas', genuine full clause",
        sentence=(
            "The appellant focused on procedural fairness, whereas the tribunal "
            "considered substantive merit the more pressing issue."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="First connective-category case built on 'whereas' -- checks it isn't "
             "mishandled differently from 'although'/'while'.",
    ),
    Case(
        name="[ADV3] 'whereas', reduced aside",
        sentence="The claimant's conduct, whereas unusual, did not breach any policy.",
        sources=[TRIBUNAL_SERVICE],
        expect_split=False,
        note="'unusual' has no subject of its own -- same reduced-aside pattern as "
             "'although serious,', now on 'whereas'.",
    ),
    Case(
        name="[ADV3] 'though', genuine full clause",
        sentence=(
            "The tribunal upheld the dismissal, though the claimant maintained his "
            "innocence throughout."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="Same coverage check for 'though'.",
    ),
    Case(
        name="[ADV3b] bare-noun subject WITH a comma soon after (real gap)",
        sentence=(
            "The claimant withdrew the application, although government policy, "
            "introduced only the previous year, shifted considerably during the "
            "appeal process."
        ),
        sources=[TRIBUNAL_SERVICE],
        expect_split=True,
        note="The first round-3 case passed by accident -- it had no comma at all after "
             "'although', so the 'runs to sentence end' fallback let it through before "
             "the subject check ever ran. THIS version adds a nested appositive "
             "('introduced only the previous year') right after the bare-noun subject, "
             "putting a comma where the subject-check actually fires. Predicted: "
             "'government' is lowercase and not in _SUBJECT_STARTERS, so this genuine "
             "clause gets wrongly treated as a subjectless aside and the split is missed.",
    ),
]


_AUX_OR_VERB_HINTS = frozenset(
    "is are was were be been being has have had do does did will would can could may "
    "might must shall should applies amounts justifies provides requires dismissed "
    "found held ruled failed succeeded reviewed considered departed amount".split()
)


def _looks_incomplete(unit: str) -> bool:
    """Crude, deliberately conservative heuristic for the adversarial report only: flags
    a unit that has no recognisable finite-verb/auxiliary word AND is short enough that
    it's plausibly a dangling subject or predicate fragment rather than a real claim.
    Not a grammar checker -- just enough to flag the cases worth reading by eye."""
    words = unit.lower().split()
    return len(words) <= 6 and not any(w.strip(".,;:") in _AUX_OR_VERB_HINTS for w in words)


def _run_case(case: Case, splitter=split_into_claims, label: str = "") -> dict:
    norm_sources = normalize_sources(case.sources)
    idf = build_idf(case.sources)

    whole_tier, whole_score = classify(case.sentence, norm_sources, idf)
    units_text = splitter(case.sentence)
    did_split = len(units_text) > 1

    header = f"{BOLD}{case.name}{RESET}" + (f"  {DIM}[{label}]{RESET}" if label else "")
    print(header)
    print(f"  {DIM}{case.note}{RESET}")
    print(f"  sentence: {case.sentence}")
    print(f"  whole-sentence tier: {_tag(whole_tier)}"
          + (f"  {DIM}C(S)={whole_score:.2f}{RESET}" if whole_score is not None else ""))

    diverged = False
    incomplete_units = 0
    if did_split:
        print(f"  split into {len(units_text)} unit(s):")
        tiers = set()
        for u in units_text:
            tier, score = classify(u, norm_sources, idf)
            tiers.add(tier)
            score_s = f"  {DIM}C(S)={score:.2f}{RESET}" if score is not None else ""
            suspect = _looks_incomplete(u)
            flag = f"  {RED}[looks incomplete]{RESET}" if suspect else ""
            if suspect:
                incomplete_units += 1
            print(f"    {_tag(tier)}{score_s}  | {u}{flag}")
        if tiers != {whole_tier}:
            print(f"  {YELLOW}>> DIVERGES from whole-sentence tier{RESET}")
            diverged = True
    else:
        print(f"  {DIM}(no connective boundary found -- scored as one unit){RESET}")

    unexpected = did_split and not case.expect_split
    missed = not did_split and case.expect_split
    if unexpected:
        print(f"  {RED}>> UNEXPECTED: splitter fired where it should not have{RESET}")
    if missed:
        print(f"  {RED}>> MISSED: splitter should have found a boundary here{RESET}")
    if incomplete_units:
        print(f"  {YELLOW}>> {incomplete_units} unit(s) look like dangling fragments, "
              f"not standalone claims{RESET}")
    print()
    return {
        "diverged": diverged,
        "unexpected": unexpected,
        "missed": missed,
        "incomplete_units": incomplete_units,
    }


def _summarize(label: str, results: List[dict], n: int) -> None:
    print(f"{CYAN}{BOLD}Summary -- {label}{RESET}")
    print(f"  cases run: {n}")
    print(f"  tier divergence from splitting: {sum(r['diverged'] for r in results)}")
    print(f"  unexpected splits: {sum(r['unexpected'] for r in results)}")
    print(f"  missed splits: {sum(r['missed'] for r in results)}")
    print(f"  cases producing >=1 dangling/incomplete-looking fragment: "
          f"{sum(1 for r in results if r['incomplete_units'])}")


def run() -> None:
    print(f"{CYAN}{BOLD}########## UNGUARDED split_into_claims ##########{RESET}\n")

    print(f"{CYAN}{BOLD}=== Realistic cases ==={RESET}\n")
    realistic_unguarded = [_run_case(c, split_into_claims, "unguarded") for c in CASES]

    print(f"{CYAN}{BOLD}=== Adversarial cases ==={RESET}\n")
    adversarial_unguarded = [_run_case(c, split_into_claims, "unguarded") for c in ADVERSARIAL_CASES]

    print(f"{CYAN}{BOLD}########## GUARDED split_into_claims_guarded ##########{RESET}\n")

    print(f"{CYAN}{BOLD}=== Realistic cases ==={RESET}\n")
    realistic_guarded = [_run_case(c, split_into_claims_guarded, "guarded") for c in CASES]

    print(f"{CYAN}{BOLD}=== Adversarial cases ==={RESET}\n")
    adversarial_guarded = [_run_case(c, split_into_claims_guarded, "guarded") for c in ADVERSARIAL_CASES]

    print(f"{CYAN}{BOLD}=== Round 2: stress-testing the guard itself ==={RESET}\n")
    round2_guarded = [_run_case(c, split_into_claims_guarded, "guarded") for c in ADVERSARIAL_ROUND2]

    print(f"{CYAN}{BOLD}=== Round 3: subject-heuristic blind spot + whereas/though coverage ==={RESET}\n")
    round3_guarded = [_run_case(c, split_into_claims_guarded, "guarded") for c in ADVERSARIAL_ROUND3]

    _summarize("realistic, unguarded", realistic_unguarded, len(CASES))
    print()
    _summarize("realistic, guarded", realistic_guarded, len(CASES))
    print()
    _summarize("adversarial, unguarded", adversarial_unguarded, len(ADVERSARIAL_CASES))
    print()
    _summarize("adversarial, guarded", adversarial_guarded, len(ADVERSARIAL_CASES))
    print()
    _summarize("round 2, guarded", round2_guarded, len(ADVERSARIAL_ROUND2))
    print()
    _summarize("round 3, guarded", round3_guarded, len(ADVERSARIAL_ROUND3))


if __name__ == "__main__":
    run()
