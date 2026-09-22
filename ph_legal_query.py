"""PH-only query planning for search_republic_acts (resources/functions/user_functions.py).

juris.ph does keyword-ish semantic matching, so acronyms ("VAWC") return nothing and the
*format* of an RA number ("RA 9262" vs "R.A. No. 9262" vs "9262") changes whether the right
act ranks at all. plan_ra_query turns what the user typed into the set of queries worth
running against juris_mcp.

Ported from ilovelawyer-api's src/utils/ph-legal-query.ts (FORHU/ilovelawyer-api#93) — same
alias table and number handling, kept as a separate copy per that ticket and FORHU/chat-wonder-v2-api#75
("consider sharing the lookup table... rather than maintaining two copies" was decided against:
different language/runtime, so two small verified copies beat one shared dependency). Update
both if the table changes. UK has its own path and must not use this module.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class _Alias:
    ra: str
    title: str


# Keys are normalized (see _alias_key). Every entry was checked against the juris.ph record
# for that RA number; add new ones the same way rather than from memory.
PH_LAW_ALIASES: dict[str, _Alias] = {
    "VAWC": _Alias("9262", "Anti-Violence Against Women and Their Children Act of 2004"),
    "IPRA": _Alias("8371", "Indigenous Peoples Rights Act of 1997"),
    "EPIRA": _Alias("9136", "Electric Power Industry Reform Act of 2001"),
    "AMLA": _Alias("9160", "Anti-Money Laundering Act of 2001"),
    "ADR": _Alias("9285", "Alternative Dispute Resolution Act of 2004"),
    "ARTA": _Alias("11032", "Ease of Doing Business and Efficient Government Service Delivery Act of 2018"),
    "EODB": _Alias("11032", "Ease of Doing Business and Efficient Government Service Delivery Act of 2018"),
    "DPA": _Alias("10173", "Data Privacy Act of 2012"),
    "CARL": _Alias("6657", "Comprehensive Agrarian Reform Law of 1988"),
    "CARP": _Alias("6657", "Comprehensive Agrarian Reform Law of 1988"),
    "CARPER": _Alias("9700", "Comprehensive Agrarian Reform Program Extension with Reforms"),
    "JJWA": _Alias("9344", "Juvenile Justice and Welfare Act of 2006"),
    "ATA": _Alias("11479", "Anti-Terrorism Act of 2020"),
    "HSA": _Alias("9372", "Human Security Act of 2007"),
    "CPA": _Alias("10175", "Cybercrime Prevention Act of 2012"),
    "SSA": _Alias("11313", "Safe Spaces Act"),
    "TRAIN": _Alias("10963", "Tax Reform for Acceleration and Inclusion"),
    "CREATE": _Alias("11534", "Corporate Recovery and Tax Incentives for Enterprises Act"),
    "CDDA": _Alias("9165", "Comprehensive Dangerous Drugs Act of 2002"),
    "ATIP": _Alias("9208", "Anti-Trafficking in Persons Act of 2003"),
    "OSAEC": _Alias("11930", "Anti-Online Sexual Abuse or Exploitation of Children"),
    "MCW": _Alias("9710", "Magna Carta of Women"),
    "LGC": _Alias("7160", "Local Government Code of 1991"),
    "RCC": _Alias("11232", "Revised Corporation Code of the Philippines"),
    "SRC": _Alias("8799", "Securities Regulation Code"),
    "GPRA": _Alias("9184", "Government Procurement Reform Act"),
    "PCA": _Alias("10667", "Philippine Competition Act"),
    "IPC": _Alias("8293", "Intellectual Property Code of the Philippines"),
    "CMTA": _Alias("10863", "Customs Modernization and Tariff Act"),
    "ESWMA": _Alias("9003", "Ecological Solid Waste Management Act of 2000"),
    "UHC": _Alias("11223", "Universal Health Care Act"),
    "RH": _Alias("10354", "Responsible Parenthood and Reproductive Health Act of 2012"),
    "RHLAW": _Alias("10354", "Responsible Parenthood and Reproductive Health Act of 2012"),
    "RTL": _Alias("11203", "Rice Tariffication Law"),
    "AFASA": _Alias("12010", "Anti-Financial Account Scamming Act"),
    "CCESPO": _Alias("6713", "Code of Conduct and Ethical Standards for Public Officials and Employees"),
    "RESA": _Alias("9646", "Real Estate Service Act of 2009"),
    "FIA": _Alias("7042", "Foreign Investments Act of 1991"),
    "BOT": _Alias("6957", "Build-Operate-Transfer Law"),
    "SEZ": _Alias("7916", "Special Economic Zone Act of 1995"),
}

_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


def _alias_key(q: str) -> str:
    """'V.A.W.C.', 'vawc', 'RH Law' -> 'VAWC', 'RHLAW'."""
    return _NON_ALNUM_RE.sub("", q.upper())


# "RA 9262", "R.A. No. 9262", "Republic Act #9262", "Rep. Act No.9262", "RA9262"
_RA_NUMBER_RE = re.compile(r"^(?:r\.?\s*a\.?|rep(?:ublic)?\.?\s*act)\s*(?:nos?\.?|number|#)?\s*(\d{1,6})$", re.IGNORECASE)
# A bare number, optionally "No. 9262" / "#9262" — 3+ digits so "5" isn't treated as a law.
_BARE_NUMBER_RE = re.compile(r"^(?:nos?\.?\s*|#\s*)?(\d{3,7})$", re.IGNORECASE)

# Issuances juris.ph doesn't index (its datasets are jurisprudence + republic-acts only).
_UNINDEXED_RE = re.compile(
    r"^(e\.?\s*o\.?|executive\s+order|p\.?\s*d\.?|presidential\s+decree|a\.?\s*o\.?|administrative\s+order|"
    r"m\.?\s*o\.?|memorandum\s+order|m\.?\s*c\.?|memorandum\s+circular|b\.?\s*p\.?(?:\s*blg\.?)?|"
    r"batas\s+pambansa(?:\s*blg\.?)?|c\.?\s*a\.?|commonwealth\s+act|act)\s*(?:nos?\.?|number|#)?\s*"
    r"(\d{1,5}(?:-[a-z])?)$",
    re.IGNORECASE,
)


def _unindexed_label(kind: str, num: str) -> str:
    t = re.sub(r"[.\s]", "", kind.lower())
    if t in ("eo", "executiveorder"):
        name = "Executive Order No."
    elif t in ("pd", "presidentialdecree"):
        name = "Presidential Decree No."
    elif t in ("ao", "administrativeorder"):
        name = "Administrative Order No."
    elif t in ("mo", "memorandumorder"):
        name = "Memorandum Order No."
    elif t in ("mc", "memorandumcircular"):
        name = "Memorandum Circular No."
    elif t.startswith("bp") or t.startswith("batas"):
        name = "Batas Pambansa Blg."
    elif t in ("ca", "commonwealthact"):
        name = "Commonwealth Act No."
    else:
        name = "Act No."
    return f"{name} {num.upper()}"


def _ra_variants(n: str) -> list[str]:
    return [f"Republic Act No. {n}", f"R.A. No. {n}", f"RA {n}"]


def _unique(items: list[str]) -> list[str]:
    seen: dict[str, None] = {}
    for item in items:
        seen.setdefault(item, None)
    return list(seen.keys())


@dataclass
class RaSearchPlan:
    """What plan_ra_query decided to do with a typed query.

    kind == "unindexed": `label` is the canonical citation (e.g. "Presidential Decree No. 442");
    the caller should return an explanatory result rather than searching.
    kind == "search": `queries` are the query strings to send to juris_mcp's search_republic_acts,
    in parallel; `ra_number` is set when this is a lookup by RA number/acronym, so the caller can
    put the exact-number hit first when merging results.
    """

    kind: str
    label: Optional[str] = None
    queries: list[str] = field(default_factory=list)
    ra_number: Optional[str] = None


def plan_ra_query(raw_query: str) -> RaSearchPlan:
    """Decides what to run for a typed query. Anything not recognised is returned untouched
    (one query), so existing behaviour is unchanged for ordinary searches. Recognition is
    whole-query on purpose: "vawc" expands, "vawc protection order" does not.
    """
    q = (raw_query or "").strip()
    plain = RaSearchPlan(kind="search", queries=[q])
    if not q:
        return plain

    issuance = _UNINDEXED_RE.match(q)
    if issuance:
        return RaSearchPlan(kind="unindexed", label=_unindexed_label(issuance.group(1), issuance.group(2)))

    ra_match = _RA_NUMBER_RE.match(q)
    bare_match = _BARE_NUMBER_RE.match(q)
    ra_number = ra_match.group(1) if ra_match else (bare_match.group(1) if bare_match else None)
    alias = PH_LAW_ALIASES.get(_alias_key(q))

    if ra_number:
        return RaSearchPlan(kind="search", queries=_ra_variants(ra_number), ra_number=ra_number)
    if alias:
        # The acronym itself is left out: on juris.ph it returns nothing or unrelated acts.
        return RaSearchPlan(
            kind="search",
            queries=_unique([*_ra_variants(alias.ra), alias.title]),
            ra_number=alias.ra,
        )
    return plain


def document_number(v: Optional[str]) -> str:
    """The RA number inside a stored/returned reference — "RA 9262", "No. 8371" and "9262" all
    compare on their first run of digits. First run only, so an amending act's "R.A. No. 9208
    (as amended by R.A. No. 10364)" still reads as 9208."""
    m = re.search(r"\d+", v or "")
    return m.group(0) if m else ""
