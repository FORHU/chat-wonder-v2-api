# -*- coding: utf-8 -*-
"""Dynamic legal turn prep: general protocol for ANY fact pattern + optional prefetch."""

from __future__ import annotations

import logging
import re
from typing import List, Optional, Tuple

# Always injected for every [legal ai] fact-pattern turn (not tied to Q1–Q5).
_GENERAL_PROTOCOL = """[LEGAL_ANALYSIS_PROTOCOL — apply to THIS question, whatever the topic]
1. Issue-spot: list every distinct legal question the user asked (a/b/c, and/or, separately…).
2. For EACH issue, derive a targeted search query (doctrine + key facts). Run multiple
   `search_jurisprudence` / `search_republic_acts` calls as needed — never one vague search for everything.
   When a doctrine has a line of cases, fetch several supporting decisions (not only one).
3. Call `get_case` / `get_republic_act` on every document you will rely on before stating holdings.
4. Write an AnyCase-style memo:
   - Opening direct answer (plain yes/no + conditions + practical prerequisite), then authority trail.
   - One H3 per user issue mirroring (a)/(b)/(c); nest numbered steps with Why / Core requirements /
     Important note / Key point / Bottom line where the user needs a path.
   - Direct-answer reprise for anxious yes/no asks; end with ONE determinative clarifying question.
5. Cite with exact juris.ph `url` links after the sentences they support; prefer paraphrase over fake quotes.
6. Never invent quotes, G.R. numbers, RA numbers, juris.ph URLs, BIR rulings, or LGU guidelines.
   Label unscourceable practice tips as general information to verify with counsel.
7. Never close all remedies ("you cannot file") without checking parallel civil/admin/labor paths.
8. If tools miss, say so — do not fill from memory as if sourced."""

# Soft hints only — accelerate common doctrines; protocol above still governs ALL topics.
_HINTS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(r"foreign divorce|remarry|japan(?:ese)?.{0,20}divorce|recognition of.{0,20}divorce", re.I),
        "Hint: search/fetch Republic v. Manalo, Ordaneza v. Republic, Bayog-Saito, and Kikuchi if found. "
        "Art. 26(2): recognition available even if Filipino filed; prove foreign divorce + foreign law; "
        "judicial recognition + civil-registry annotation before remarriage; recognition ≠ automatic property award.",
    ),
    (
        re.compile(r"floating status|off-detail|security guard|temporary off", re.I),
        "Hint: search floating status six months constructive dismissal; resignation is NOT required; cover prescription + NLRC/SEnA.",
    ),
    (
        re.compile(r"one person corporation|\bOPC\b|sole stockholder|nominee|pierce.{0,20}veil", re.I),
        "Hint: RA 11232 Sec. 130 burden reversal; nominee gap ≠ automatic personal liability; RTC Special Commercial Court not SEC.",
    ),
    (
        re.compile(r"free patent|double sale|unregistered sale|torrens|foreign heir|naturalized.{0,30}(inherit|heir|land)", re.I),
        "Hint: Art. 1544 / Torrens direct action; RA 11231 (ag) vs RA 10023/RA 730 (residential); hereditary succession Art. XII Sec. 7.",
    ),
    (
        re.compile(r"cyber libel|online libel|defamat|facebook.{0,20}post", re.I),
        "Hint: Causing v. People (often 1-year criminal prescription); always consider Art. 33 independent civil action.",
    ),
    (
        re.compile(r"\bVAT\b|digital (services|subscription)|zero[\s-]?rat", re.I),
        "Hint: search RA 12023 digital services VAT; foreign customers may implicate zero-rating — do not stop at TRAIN alone.",
    ),
    (
        re.compile(r"illegal dismissal|constructive dismissal|backwages|separation pay|NLRC", re.I),
        "Hint: search controlling dismissal doctrine + Art. 294/279 framing; flag 4-year vs 3-year prescription when money claims appear.",
    ),
    (
        re.compile(r"annulment|legal separation|custody|support|Family Code", re.I),
        "Hint: search on-point SC family-law cases; do not invent Family Code article quotes without get_* text.",
    ),
]

# Keyword → likely RA to prefetch when user did not name a number (soft, capped).
_KEYWORD_RA: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"one person corporation|\bOPC\b", re.I), "RA 11232"),
    (re.compile(r"agricultural free patent|free patent", re.I), "RA 11231"),
    (re.compile(r"residential free patent", re.I), "RA 10023"),
    (re.compile(r"cybercrime|cyber libel", re.I), "RA 10175"),
    (re.compile(r"digital services.{0,40}VAT|VAT.{0,40}digital", re.I), "RA 12023"),
    (re.compile(r"safe spaces|gender.?based.{0,20}harassment", re.I), "RA 11313"),
    (re.compile(r"data privacy", re.I), "RA 10173"),
    (re.compile(r"anti.?violence against women|\bVAWC\b", re.I), "RA 9262"),
]

_MAX_PREFETCH = 4


def apply_legal_fact_pattern_boost(user_input: str) -> str:
    """Always attach the general protocol; add soft doctrine hints when keywords match."""
    text = (user_input or "").strip()
    if not text:
        return user_input

    lines = [_GENERAL_PROTOCOL]
    hints: List[str] = []
    seen = set()
    for pattern, hint in _HINTS:
        if pattern.search(text) and hint not in seen:
            hints.append(f"- {hint}")
            seen.add(hint)
    if hints:
        lines.append("[DOCTRINE_HINTS — optional accelerators; still run tools and follow the protocol]")
        lines.extend(hints)

    return f"{text}\n\n" + "\n".join(lines)


def _extract_explicit_specs(text: str) -> List[dict]:
    """Pull RA / G.R. numbers the user (or prior context) already named."""
    specs: List[dict] = []
    seen = set()

    for m in re.finditer(
        r"(?:RA|R\.A\.|Republic Act)\s*(?:No\.?\s*)?(\d{3,5})\b",
        text,
        flags=re.I,
    ):
        ra = f"RA {m.group(1)}"
        key = ("ra", ra)
        if key not in seen:
            seen.add(key)
            specs.append({"kind": "ra", "ra_number": ra})

    for m in re.finditer(r"G\.?\s*R\.?\s*No\.?\s*([\d-]+)", text, flags=re.I):
        gr = f"G.R. No. {m.group(1)}"
        key = ("case", gr)
        if key not in seen:
            seen.add(key)
            specs.append({"kind": "case", "case_number": gr})

    return specs


def _keyword_ra_specs(text: str) -> List[dict]:
    specs: List[dict] = []
    seen = set()
    for pattern, ra in _KEYWORD_RA:
        if pattern.search(text) and ra not in seen:
            seen.add(ra)
            specs.append({"kind": "ra", "ra_number": ra})
    return specs


def _entry_from_get(result: dict, *, type_hint: str) -> Optional[dict]:
    if not isinstance(result, dict) or not result.get("success"):
        return None
    doc = result.get("document") if isinstance(result.get("document"), dict) else {}
    summary = ""
    if doc:
        summary = (
            doc.get("summary")
            or doc.get("court_reasoning")
            or doc.get("factual_background")
            or ""
        )
    return {
        "id": result.get("id") or result.get("item_id"),
        "item_id": result.get("item_id") or result.get("id"),
        "title": result.get("title") or (doc.get("title") if doc else ""),
        "url": result.get("url"),
        "type": result.get("type") or type_hint,
        "year": result.get("year") or doc.get("year"),
        "snippet": summary[:1200] if summary else "",
        "disposition": doc.get("final_disposition") or doc.get("disposition") or "",
        "court_reasoning": doc.get("court_reasoning") or "",
        "document": doc or result.get("document"),
        "prefetched": True,
    }


def prefetch_legal_authorities(user_input: str) -> Tuple[List[dict], str]:
    """
    Dynamic prefetch: explicit RA/G.R. in the query first, then keyword RA hints.
    Works for any topic; not limited to a fixed eval list.
    """
    text = (user_input or "").strip()
    if not text:
        return [], ""

    specs: List[dict] = []
    seen = set()
    for spec in _extract_explicit_specs(text) + _keyword_ra_specs(text):
        key = (spec.get("kind"), spec.get("ra_number"), spec.get("case_number"))
        if key in seen:
            continue
        seen.add(key)
        specs.append(spec)
        if len(specs) >= _MAX_PREFETCH:
            break

    if not specs:
        return [], ""

    from resources.functions import user_functions as uf

    entries: List[dict] = []
    lines: List[str] = []
    for spec in specs:
        try:
            if spec.get("kind") == "ra":
                result = uf.get_republic_act(ra_number=spec["ra_number"])
                entry = _entry_from_get(result, type_hint="republic_act")
                label = spec["ra_number"]
            else:
                result = uf.get_case(case_number=spec["case_number"])
                entry = _entry_from_get(result, type_hint="jurisprudence")
                label = spec["case_number"]
            if not entry or not entry.get("url"):
                logging.warning("[legal-prefetch] miss for %s: %s", label, (result or {}).get("error"))
                continue
            entries.append(entry)
            snip = (entry.get("snippet") or "")[:400]
            lines.append(
                f"- {entry.get('title') or label} | {entry.get('url')}"
                + (f"\n  Summary: {snip}" if snip else "")
            )
        except Exception as e:
            logging.warning("[legal-prefetch] failed %s: %s", spec, e)

    if not entries:
        return [], ""

    injection = (
        "[PREFETCHED_AUTHORITIES — already retrieved from the user's topic signals; "
        "cite these juris.ph URLs when on-point; still search for any other controlling authority]\n"
        + "\n".join(lines)
    )
    return entries, injection


def prepare_legal_turn(user_input: str) -> Tuple[str, List[dict]]:
    """General protocol + dynamic prefetch for any legal question."""
    boosted = apply_legal_fact_pattern_boost(user_input)
    entries, injection = prefetch_legal_authorities(user_input)
    if injection:
        boosted = f"{boosted.rstrip()}\n\n{injection}"
    return boosted, entries


def append_critical_doctrine_guards(text: str, user_input: str) -> str:
    """
    Light, topic-triggered reminders — never claim to be exhaustive.
    Prefer model tool use; these only catch common omissions.
    """
    if not text:
        return text
    u = user_input or ""
    additions: List[str] = []

    if re.search(r"cyber libel|online libel|defamat", u, re.I) and not re.search(
        r"Article\s*33|Art\.\s*33|independent civil", text, re.I
    ):
        additions.append(
            "- **Parallel civil remedy:** If a criminal defamation/cyber libel path looks time-barred, "
            "consider an **independent civil action under Civil Code Article 33** (often a longer "
            "prescription) before concluding nothing can be filed — verify with counsel."
        )

    if re.search(r"free patent|double sale|torrens|unregistered sale", u, re.I):
        if re.search(r"free patent", u, re.I) and not re.search(r"11231|10023|residential|agricultural", text, re.I):
            additions.append(
                "- **Patent-type fork:** Confirm agricultural free patent (**RA 11231** / CA 141) vs "
                "residential (**RA 10023** / older schemes) — restrictions and alienability can differ."
            )
        if not re.search(r"direct action|reconvey|Torrens|good faith", text, re.I):
            additions.append(
                "- **Title litigation:** Registered Torrens titles are usually challenged by "
                "**direct action** (reconveyance/annulment), with good faith often decisive in double sales."
            )

    if re.search(r"naturalized|foreign.{0,20}heir|american citizen.{0,40}inherit", u, re.I) and not re.search(
        r"hereditary|Article\s*XII|Art\.\s*XII", text, re.I
    ):
        additions.append(
            "- **Foreign heir:** Land acquisition by **hereditary succession** is the usual "
            "constitutional exception (Art. XII Sec. 7) — citizenship alone does not always bar inheritance."
        )

    if re.search(r"floating status|off-detail", u, re.I) and not re.search(
        r"six[\s-]?month|6[\s-]?month", text, re.I
    ):
        additions.append(
            "- **Floating / off-detail:** Security-agency temporary off-detail is commonly analyzed "
            "under a **six-month** outer limit; longer limbo may be treated as constructive dismissal "
            "without requiring a resignation — verify with current jurisprudence."
        )

    if not additions:
        return text
    block = "\n\n## Do not overlook\n" + "\n".join(additions) + "\n"
    marker = "[RELATED_QUERIES]"
    if marker in text:
        return text.replace(marker, block + marker, 1)
    disclaimer = "For your specific situation, please consult a licensed attorney."
    if disclaimer in text:
        return text.replace(disclaimer, block + "\n" + disclaimer, 1)
    return text.rstrip() + block


def append_missing_prefetched_mentions(text: str, search_results) -> str:
    """If a prefetched authority was retrieved but never named, append a short cite block."""
    if not text or not search_results:
        return text
    additions: List[str] = []
    for row in search_results:
        if not isinstance(row, dict) or not row.get("prefetched"):
            continue
        title = str(row.get("title") or "")
        url = str(row.get("url") or "").strip()
        snippet = str(row.get("snippet") or "")
        blob = f"{title} {snippet}"
        ra_nums = re.findall(r"(?:RA|Republic Act)\s*No\.?\s*(\d{4,5})", blob, flags=re.I)
        if not ra_nums:
            ra_nums = re.findall(r"\b(\d{4,5})\b", title)
        gr = re.search(r"G\.R\.\s*No\.\s*[\d-]+", blob, flags=re.I)

        mentioned = False
        for n in ra_nums:
            if re.search(rf"\b{re.escape(n)}\b", text):
                mentioned = True
                break
        if gr and gr.group(0).lower().replace(" ", "") in re.sub(r"\s+", "", text.lower()):
            mentioned = True
        if title and len(title) > 12 and title.lower() in text.lower():
            mentioned = True
        if mentioned or not url:
            continue

        label = title or (f"RA {ra_nums[0]}" if ra_nums else "Retrieved authority")
        if ra_nums and f"RA {ra_nums[0]}" not in label and ra_nums[0] not in label:
            label = f"{label} (RA {ra_nums[0]})"
        suffix = " Law" if (row.get("type") == "republic_act" or ra_nums) else " Jurisprudence"
        if not label.endswith(suffix.strip()):
            link_text = f"{label}{suffix}"
        else:
            link_text = label
        snip = snippet.strip()
        if len(snip) > 280:
            snip = snip[:277] + "..."
        additions.append(f"- [{link_text}]({url})" + (f" — {snip}" if snip else ""))

    if not additions:
        return text
    block = (
        "\n\n## Controlling authorities retrieved for this fact pattern\n"
        + "\n".join(additions)
        + "\n"
    )
    marker = "[RELATED_QUERIES]"
    if marker in text:
        return text.replace(marker, block + marker, 1)
    disclaimer = "For your specific situation, please consult a licensed attorney."
    if disclaimer in text:
        return text.replace(disclaimer, block + "\n" + disclaimer, 1)
    return text.rstrip() + block
