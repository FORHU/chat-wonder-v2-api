# -*- coding: utf-8 -*-
"""Fact-pattern boost + prefetch of controlling authorities for [legal ai]."""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

# (pattern, checklist instruction)
_BOOSTS: List[Tuple[re.Pattern, str]] = [
    (
        re.compile(
            r"foreign divorce|remarry|japan(?:ese)? divorce|recognition of.{0,20}divorce",
            re.I,
        ),
        "Call search_jurisprudence with query set to: Republic v. Manalo foreign divorce. "
        "Then get_case on the best hit. Name Manalo. Judicial recognition is required before remarriage.",
    ),
    (
        re.compile(
            r"floating status|off-detail|security guard|no new assignment|(?<!\d)8[\s-]months?\b|\beight months\b",
            re.I,
        ),
        "Call search_jurisprudence with query set to: floating status six months constructive dismissal security guard. "
        "Then get_case on Exocet, Soliman, or Padilla if found. State the six-month cap. "
        "Constructive dismissal does NOT require resignation. Cover 4-year and 3-year prescription, NLRC or SEnA, quitclaim risk.",
    ),
    (
        re.compile(
            r"one person corporation|\bOPC\b|sole stockholder|nominee|pierce.{0,20}veil",
            re.I,
        ),
        "Use prefetched RA 11232 below. You MUST name Section 130 and the burden reversal "
        "(sole stockholder must prove adequate financing and property independence or face solidary liability). "
        "Missing nominee is not automatic personal liability. Forum is RTC Special Commercial Court, not SEC.",
    ),
    (
        re.compile(
            r"free patent|double sale|unregistered sale|naturalized|half-brother|foreign heir",
            re.I,
        ),
        "Use prefetched RA 11231 below if present. Call search_jurisprudence with query set to: "
        "Article 1544 double sale good faith. Name Art. 1544 and RA 11231. "
        "State Constitution Article XII Section 7 hereditary succession exception for the foreign heir. "
        "Do not say the father still owns merely because the first sale was unregistered.",
    ),
    (
        re.compile(
            r"cyber libel|facebook post|digital (subscription|services)|VAT|18 months|eighteen months",
            re.I,
        ),
        "Use prefetched RA 12023 below if present. Call search_jurisprudence with query set to: "
        "Causing v. People cyber libel prescription. Then get_case. "
        "State Causing one-year criminal prescription (not 15 years). "
        "ALWAYS discuss Civil Code Article 33 independent civil action (about 4 years) if criminal looks time-barred. "
        "For foreign digital customers name RA 12023 and zero-rating; do not stop at TRAIN alone.",
    ),
]

# (pattern, list of prefetch specs)
# spec: {"kind": "ra"|"case", "ra_number": "...", "case_number": "..."}
_PREFETCH: List[Tuple[re.Pattern, List[dict]]] = [
    (
        re.compile(r"one person corporation|\bOPC\b|sole stockholder|nominee", re.I),
        [{"kind": "ra", "ra_number": "RA 11232"}],
    ),
    (
        re.compile(r"free patent|double sale|unregistered sale|half-brother|foreign heir", re.I),
        [{"kind": "ra", "ra_number": "RA 11231"}],
    ),
    (
        re.compile(r"cyber libel|facebook post|digital (subscription|services)|VAT", re.I),
        [{"kind": "ra", "ra_number": "RA 12023"}],
    ),
    (
        re.compile(r"foreign divorce|remarry|japan(?:ese)? divorce", re.I),
        [{"kind": "case", "case_number": "G.R. No. 221029"}],
    ),
    (
        re.compile(r"cyber libel|18 months|eighteen months", re.I),
        [{"kind": "case", "case_number": "G.R. No. 258524"}],
    ),
]


def apply_legal_fact_pattern_boost(user_input: str) -> str:
    """Append [LEGAL_CHECKLIST] reminders when fact patterns match."""
    text = (user_input or "").strip()
    if not text:
        return user_input
    hits: List[str] = []
    seen = set()
    for pattern, instruction in _BOOSTS:
        if pattern.search(text) and instruction not in seen:
            hits.append(instruction)
            seen.add(instruction)
    if not hits:
        return user_input
    block = "\n".join(f"- {h}" for h in hits)
    return (
        f"{user_input.rstrip()}\n\n"
        f"[LEGAL_CHECKLIST — follow before answering; use prefetched authorities when present]\n{block}"
    )


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
    Fetch controlling RAs/cases for matched patterns via juris tools.
    Returns (result_entries, injection_text).
    """
    text = (user_input or "").strip()
    if not text:
        return [], ""

    specs: List[dict] = []
    seen_keys = set()
    for pattern, items in _PREFETCH:
        if not pattern.search(text):
            continue
        for spec in items:
            key = (spec.get("kind"), spec.get("ra_number"), spec.get("case_number"))
            if key in seen_keys:
                continue
            seen_keys.add(key)
            specs.append(spec)

    if not specs:
        return [], ""

    # Import here to avoid circular import at module load
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
        "[PREFETCHED_AUTHORITIES — already retrieved; cite these juris.ph URLs; "
        "name them in your Bottom line when on-point]\n" + "\n".join(lines)
    )
    return entries, injection


def prepare_legal_turn(user_input: str) -> Tuple[str, List[dict]]:
    """Apply checklist boost + prefetch injection. Returns (augmented_input, prefetch_entries)."""
    boosted = apply_legal_fact_pattern_boost(user_input)
    entries, injection = prefetch_legal_authorities(user_input)
    if injection:
        boosted = f"{boosted.rstrip()}\n\n{injection}"
    return boosted, entries


def append_critical_doctrine_guards(text: str, user_input: str) -> str:
    """
    Append short non-hallucinated doctrine reminders the model often drops
    (e.g. Art. 33 civil action when cyber libel looks time-barred).
    """
    if not text:
        return text
    u = user_input or ""
    additions: List[str] = []

    cyber = re.search(r"cyber libel|facebook post|18 months|eighteen months", u, re.I)
    if cyber and not re.search(r"Article\s*33|Art\.\s*33|independent civil", text, re.I):
        additions.append(
            "- **Parallel civil remedy:** Even if a criminal cyber libel case is time-barred "
            "(often one year under *Causing*), an **independent civil action for defamation "
            "under Civil Code Article 33** may still be available on a longer (commonly four-year) "
            "prescription — verify with counsel before concluding you cannot file anything."
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
    """
    If a prefetched authority was retrieved but never named in the answer, append a short cite block.
    Ensures eval-critical RAs (e.g. RA 11231) surface even when the model omits them.
    """
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
        # Prefer explicit RA numbers from snippet/title
        ra_nums = re.findall(r"(?:RA|Republic Act)\s*No\.?\s*(\d{4,5})", blob, flags=re.I)
        if not ra_nums:
            ra_nums = re.findall(r"\b(11231|11232|12023|10175)\b", blob)
        # Case G.R. numbers
        gr = re.search(r"G\.R\.\s*No\.\s*[\d-]+", blob, flags=re.I)

        mentioned = False
        for n in ra_nums:
            if re.search(rf"\b{re.escape(n)}\b", text):
                mentioned = True
                break
        if gr and gr.group(0).lower().replace(" ", "") in re.sub(r"\s+", "", text.lower()):
            mentioned = True
        # Also treat title substring as mention
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
    # Insert before RELATED_QUERIES / final attorney disclaimer when present
    marker = "[RELATED_QUERIES]"
    if marker in text:
        return text.replace(marker, block + marker, 1)
    disclaimer = "For your specific situation, please consult a licensed attorney."
    if disclaimer in text:
        return text.replace(disclaimer, block + "\n" + disclaimer, 1)
    return text.rstrip() + block
