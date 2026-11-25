"""Canonical legislation.gov.uk URLs for UK Legal MCP results.

Confirmed live: `legislation_get_section` returns section text with no URL at
all, and `citations_resolve` resolves `s.94 Employment Rights Act 1996` to
`https://www.legislation.gov.uk/search?title=Employment+Rights+Act+1996` — a
search page, not the section. Left alone, every section citation in a memo
links to the same search box (19 identical hrefs in one live answer) while
still passing the cite gate, because that search URL *is* in the pool.

These helpers build the real section URL from data we already hold and are
applied by the_server.execute_function_call before the result reaches either
the model or the citation pool.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, Optional

_LEGISLATION_BASE = "https://www.legislation.gov.uk"
_SEARCH_URL = re.compile(r"^https?://(www\.)?legislation\.gov\.uk/search\b", re.I)
_ACT_URL = re.compile(r"^https?://(www\.)?legislation\.gov\.uk/(?P<type>[a-z]+)/(?P<year>\d{4})/(?P<number>\d+)/?$", re.I)


def section_url(doc_type: Any, year: Any, number: Any, section: Any) -> Optional[str]:
    """`https://www.legislation.gov.uk/ukpga/1996/18/section/94`, or None if any part is missing."""
    t = str(doc_type or "").strip().lower()
    s = str(section or "").strip()
    try:
        y, n = int(year), int(number)
    except (TypeError, ValueError):
        return None
    if not t or not s:
        return None
    return f"{_LEGISLATION_BASE}/{t}/{y}/{n}/section/{s}"


def enrich_section_result(result: Dict[str, Any], args: Dict[str, Any]) -> Dict[str, Any]:
    """Add `url` to a successful legislation_get_section result (in place) from its call args."""
    if not isinstance(result, dict) or not result.get("success") or result.get("url"):
        return result
    url = section_url(args.get("type"), args.get("year"), args.get("number"), args.get("section"))
    if url:
        result["url"] = url
    return result


def _is_search_url(url: Any) -> bool:
    return bool(url) and bool(_SEARCH_URL.match(str(url)))


def find_act_identity(pool: Iterable[Any], title: str) -> Optional[Dict[str, Any]]:
    """Locate {type, year, number} for an Act by title among earlier tool results
    (legislation_search rows carry `url` like .../ukpga/1996/18 plus type/year/number)."""
    want = re.sub(r"\s+", " ", str(title or "")).strip().lower()
    if not want:
        return None

    def walk(obj, depth=0):
        if depth > 5 or obj is None:
            return None
        if isinstance(obj, dict):
            t = re.sub(r"\s+", " ", str(obj.get("title") or "")).strip().lower()
            if t == want:
                if obj.get("type") and obj.get("year") and obj.get("number"):
                    return {"type": obj["type"], "year": obj["year"], "number": obj["number"]}
                m = _ACT_URL.match(str(obj.get("url") or ""))
                if m:
                    return {"type": m.group("type"), "year": m.group("year"), "number": m.group("number")}
            for v in obj.values():
                hit = walk(v, depth + 1)
                if hit:
                    return hit
        elif isinstance(obj, (list, tuple)):
            for item in obj[:50]:
                hit = walk(item, depth + 1)
                if hit:
                    return hit
        return None

    return walk(list(pool))


def enrich_resolve_result(result: Dict[str, Any], pool: Iterable[Any]) -> Dict[str, Any]:
    """Replace a search-page `resolved_url` on a legislation citation with the real
    section URL when the Act's identity is known from earlier results (in place).
    The original is kept under `search_url` so nothing is lost."""
    if not isinstance(result, dict) or not result.get("success"):
        return result
    if str(result.get("type") or "").lower() != "legislation":
        return result
    if not _is_search_url(result.get("resolved_url")):
        return result
    section = result.get("section")
    ident = find_act_identity(pool, result.get("legislation_title") or "")
    if not ident:
        return result
    url = (
        section_url(ident["type"], ident["year"], ident["number"], section)
        if section
        else f"{_LEGISLATION_BASE}/{str(ident['type']).lower()}/{int(ident['year'])}/{int(ident['number'])}"
    )
    if url:
        result["search_url"] = result["resolved_url"]
        result["resolved_url"] = url
    return result
