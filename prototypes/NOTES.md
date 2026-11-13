# Prototype: atomic sub-sentence claim splitting at connectives

**Question:** Dr. Kim (VeriLENS) suggested splitting compound sentences into atomic claim
units at connectives (다만/그러나/따라서 -- "however/but/therefore") before scoring, instead of
scoring a whole sentence as one unit. Does this change `provenance_spans`' classification
in cases that matter for chat-wonder's English/UK legal text -- i.e. does a compound
sentence ever hide a QUOTED clause behind an UNSOURCED one (or vice versa) when the whole
sentence is scored together?

**Files:**
- `claim_split_prototype.py` -- pure logic: `split_into_claims()` (connective-based
  splitter, English connective set) and `classify()` (calls confidence.py's real
  `_is_verbatim_match`/`_coverage_score`, not a reimplementation).
- `claim_split_tui.py` -- throwaway terminal shell. Run with:
  ```
  cd chat-wonder-v2-api
  python prototypes/claim_split_tui.py
  ```
  Type a sentence (`s`), add/remove sources (`o`/`r`), classify (`c`), and watch whether
  the whole-sentence tier matches every claim unit's tier.

**Not done here, deliberately:**
- Not wired into `confidence.py` or `_candidate_sentences`.
- No character-offset tracking (Dr. Kim's advice mentioned this; see the conversation --
  offsets into raw markdown don't survive `marked.parse()` rendering, so this would need
  to be offsets into the cleaned/rendered text, a separate design question).
- No heading/section-label exclusion (also from Dr. Kim's advice; conflicts with the
  current deliberate choice to keep headings as candidates -- not resolved here).

## Verdict

**Splitting matters and was folded into `confidence.py`.** Driven against 10 realistic
compound-sentence cases plus 14 adversarial cases across three rounds
(`claim_split_batch.py`):

1. **Confirmed the original hypothesis**: a compound sentence with a QUOTED clause
   trailing (or leading) an UNSOURCED opinion clause gets the opinion's tier masked by
   `_long_verbatim_run`'s >=50%-or-100-char rule when scored whole. Splitting at
   connectives fixes this.
2. **A naive/unguarded splitter is worse than not splitting**: it tears a parenthetical
   aside out as its own unit ("X, although Y, Z" -> wrongly confident tag on "although Y"),
   a more severe failure (confidently WRONG) than the masking bug it was meant to fix.
3. **Fixed with a subject-presence guard** (`_opens_subjectless_aside`,
   `_RISKY_SUBORDINATORS` = although/though/while/whereas/yet): a reduced aside omits its
   own subject, a genuine subordinate clause never does. Re-validated with zero regressions
   against all realistic + round-1 + round-2 adversarial cases.
   - An earlier verb-presence version of this guard was tried and broken by reduced
     past-participle asides ("although delayed,") whose "-ed" ending looks exactly like a
     real verb with no subject check -- see the superseded comment left in
     `claim_split_prototype.py` for the record.
4. **One residual gap, accepted rather than fixed**: a genuine clause whose subject is a
   bare common noun with no article ("although government policy, introduced last year,
   shifted...") is indistinguishable by this heuristic from a reduced aside, and the split
   is missed. This degrades SAFELY -- it falls back to whole-sentence scoring for that
   clause, never to a confidently wrong tier -- and closing it for real needs POS/dependency
   parsing (e.g. spaCy), out of scope for a regex-based module. Documented, not fixed.
5. **Also out of scope, lower severity**: Oxford-comma list coordination ("the contract,
   the handbook, but not the appendix.") leaves a dangling fragment after `but`; never
   observed flipping a tier, just a cosmetic leftover. `but` was left unguarded rather than
   adding a check with no observed tier-flipping failure to fix.

**Folded into `confidence.py`**: `_split_into_claim_units` (= this prototype's
`split_into_claims_guarded`), `_opens_subjectless_aside`, `_CONNECTIVES`, `_CONN_RE`,
`_RISKY_SUBORDINATORS`, `_SUBJECT_STARTERS` were lifted in as-is and wired into
`provenance_spans`, which now classifies each candidate sentence's claim units instead of
the whole sentence. This directory (prototype module + TUI + batch harness) can be deleted
once the fold-in is confirmed working against real traffic; kept for now as the record of
how the guard was arrived at.
