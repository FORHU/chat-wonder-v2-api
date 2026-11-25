"""Build draft_pleading_uk `authorities` from sources already fetched in this session.

The model is told to pass fetched authorities into the drafting call but often leaves the field
empty (#73). The session's citation pool already holds every UK Legal MCP result, so when the
model sends none, the server derives them here instead of relying on the model.

Only sources the model actually opened are used — sections read via legislation_get_section and
judgments read via judgment_get_header — never bare search hits, which would pad a pleading with
authorities nobody read.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Optional

_SECTION_URL = re.compile(
    r"^https?://(?:www\.)?legislation\.gov\.uk/(?P<type>[a-z]+)/(?P<year>\d{4})/(?P<number>\d+)/section/(?P<section>[^/?#]+)/?$",
    re.I,
)
_ACT_URL = re.compile(r"^https?://(?:www\.)?legislation\.gov\.uk/(?P<type>[a-z]+)/(?P<year>\d{4})/(?P<number>\d+)/?$", re.I)

MAX_AUTHORITIES = 10


def _walk_dicts(obj: Any, depth: int = 0):
    if depth > 6 or obj is None:
        return
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_dicts(v, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for item in obj[:100]:
            yield from _walk_dicts(item, depth + 1)


def _act_titles(pool: Iterable[Any]) -> Dict[tuple, str]:
    """(type, year, number) -> Act title, from any pool dict whose url is an Act-level url."""
    titles: Dict[tuple, str] = {}
    for d in _walk_dicts(list(pool)):
        m = _ACT_URL.match(str(d.get("url") or ""))
        title = str(d.get("title") or "").strip()
        if m and title:
            titles.setdefault((m.group("type").lower(), int(m.group("year")), int(m.group("number"))), title)
    return titles


def authorities_from_pool(pool: Optional[Iterable[Any]]) -> List[Dict[str, str]]:
    pool = list(pool or [])
    acts = _act_titles(pool)
    found: List[Dict[str, str]] = []
    seen = set()

    for entry in pool:
        if not isinstance(entry, dict):
            continue
        url = str(entry.get("url") or "")
        m = _SECTION_URL.match(url)
        if m:
            key = (m.group("type").lower(), int(m.group("year")), int(m.group("number")))
            section = m.group("section")
            act = str(entry.get("legislation_title") or entry.get("act_title") or "").strip() or acts.get(key)
            citation = f"{act}, s {section}" if act else f"legislation.gov.uk/{key[0]}/{key[1]}/{key[2]}, s {section}"
            proposition = str(entry.get("title") or "").strip().rstrip(".")
        elif isinstance(entry.get("results"), list):
            continue
        else:
            citation = str(entry.get("neutral_citation") or entry.get("citation") or "").strip()
            name = str(entry.get("case_name") or entry.get("title") or "").strip()
            if not citation or not name:
                continue
            citation = f"{name} {citation}"
            proposition = ""
        if citation in seen:
            continue
        seen.add(citation)
        found.append({"citation": citation, "proposition": proposition or "authority retrieved during research"})

    return found[-MAX_AUTHORITIES:]
