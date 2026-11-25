"""UK-only query planning for legislation_search (resources/functions/user_functions.py).

uk-legal-mcp.fly.dev's legislation_search only matches close to the literal Act title, so a
common short form ("HRA", "PACE", "DPA") returns nothing. This module resolves a recognised
acronym to its full title before the query is sent, and tags the result with a stable document
reference (legislation type/year/number, e.g. ukpga/2010/15 = legislation.gov.uk's own URI
scheme) so the caller can put the exact Act first even when an amending Act with a near-identical
title otherwise outranks it (seen for Equality Act 2010 and Mental Health Act 1983).

Separate from ph_legal_query.py on purpose (FORHU/chat-wonder-v2-api#94, UK counterpart of #75)
— different alias table, different jurisdiction, must not be shared with the PH path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class DocRef:
    """legislation.gov.uk's own identifier for one document — type/year/number, as returned by
    legislation_search and forming its url (https://www.legislation.gov.uk/{type}/{year}/{number})."""

    type: str
    year: int
    number: int


@dataclass(frozen=True)
class _Alias:
    title: str
    ref: DocRef


# Keys are normalized (see _alias_key). Every entry's `ref` was read from the live
# legislation_search record for that Act — add new ones the same way, not from memory.
UK_LAW_ALIASES: dict[str, _Alias] = {
    "HRA": _Alias("Human Rights Act 1998", DocRef("ukpga", 1998, 42)),
    "DPA": _Alias("Data Protection Act 2018", DocRef("ukpga", 2018, 12)),
    "PACE": _Alias("Police and Criminal Evidence Act 1984", DocRef("ukpga", 1984, 60)),
    "FOIA": _Alias("Freedom of Information Act 2000", DocRef("ukpga", 2000, 36)),
    "TUPE": _Alias(
        "The Transfer of Undertakings (Protection of Employment) Regulations 2006",
        DocRef("uksi", 2006, 246),
    ),
    "IHTA": _Alias("Inheritance Tax Act 1984", DocRef("ukpga", 1984, 51)),
    "SGA": _Alias("Sale of Goods Act 1979", DocRef("ukpga", 1979, 54)),
    "CRA": _Alias("Consumer Rights Act 2015", DocRef("ukpga", 2015, 15)),
    "LPA": _Alias("Law of Property Act 1925", DocRef("ukpga", 1925, 20)),
    "MHA": _Alias("Mental Health Act 1983", DocRef("ukpga", 1983, 20)),
    "ERA": _Alias("Employment Rights Act 1996", DocRef("ukpga", 1996, 18)),
    "EA": _Alias("Equality Act 2010", DocRef("ukpga", 2010, 15)),
    "CDPA": _Alias("Copyright, Designs and Patents Act 1988", DocRef("ukpga", 1988, 48)),
    "POCA": _Alias("Proceeds of Crime Act 2002", DocRef("ukpga", 2002, 29)),
    "RIPA": _Alias("Regulation of Investigatory Powers Act 2000", DocRef("ukpga", 2000, 23)),
    "CCA": _Alias("Consumer Credit Act 1974", DocRef("ukpga", 1974, 39)),
    "LASPO": _Alias(
        "Legal Aid, Sentencing and Punishment of Offenders Act 2012", DocRef("ukpga", 2012, 10)
    ),
    "FSMA": _Alias("Financial Services and Markets Act 2000", DocRef("ukpga", 2000, 8)),
    "MCA": _Alias("Mental Capacity Act 2005", DocRef("ukpga", 2005, 9)),
    "SOA": _Alias("Sexual Offences Act 2003", DocRef("ukpga", 2003, 42)),
}

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


def _alias_key(q: str) -> str:
    """'H.R.A.', 'hra' -> 'HRA'."""
    return _NON_ALNUM_RE.sub("", q.upper())


@dataclass
class UkLegislationPlan:
    """kind == "alias": `query` is the Act's full title (what actually ranks well upstream);
    `ref` is the exact document to prefer if legislation_search's own ranking puts an amending
    Act ahead of it. kind == "plain": the input passes through unchanged, as today."""

    kind: str
    query: str
    ref: Optional[DocRef] = None


def plan_uk_legislation_query(raw_query: str) -> UkLegislationPlan:
    """Recognises a whole-query acronym/short form ("HRA") and expands it to the Act's full
    title. Anything else — including an acronym inside a longer query ("HRA damages claim") —
    passes through untouched, so ordinary searches are unaffected."""
    q = (raw_query or "").strip()
    alias = UK_LAW_ALIASES.get(_alias_key(q)) if q else None
    if alias:
        return UkLegislationPlan(kind="alias", query=alias.title, ref=alias.ref)
    return UkLegislationPlan(kind="plain", query=q)


def prefer_exact_ref(rows: list, ref: DocRef) -> list:
    """Moves the row matching `ref` (type/year/number) to the front, if present — the fix for
    an amending Act (e.g. "Worker Protection (Amendment of Equality Act 2010) Act 2023")
    outranking the Act the alias actually meant."""
    if not rows:
        return rows

    def is_exact(row: dict) -> bool:
        return (
            row.get("type") == ref.type
            and row.get("year") == ref.year
            and row.get("number") == ref.number
        )

    exact = [r for r in rows if isinstance(r, dict) and is_exact(r)]
    if not exact:
        return rows
    rest = [r for r in rows if not (isinstance(r, dict) and is_exact(r))]
    return exact + rest
