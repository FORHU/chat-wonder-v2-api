"""Direct legislation.gov.uk access — the fallback for the UK legal tools when uk-legal-mcp.fly.dev cannot get through.

The MCP fetches legislation.gov.uk from its own host, and that host is currently met with "AWS WAF JavaScript
challenge" on every legislation_search / legislation_get_section call. The same public endpoints answer normally
from our own hosts (checked from a laptop and from the UK EC2 instance: 200, real XML), so these functions call
them directly and return the same shapes the MCP does — rows with `title/type/year/number/url`, and a section with
`title/section_number/content` — which is all uk_legal_mcp.urls, authorities and the citation pool read.

Endpoints (public, open data):
  search by title   {BASE}/{type|all}[/{year}]/data.feed?title=...
  search full text  {BASE}/search/data.feed?text=...[&type=...][&year=...]
  one section       {BASE}/{type}/{year}/{number}/section/{section}/data.xml
"""

from __future__ import annotations

import logging
import re
import time
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional
from urllib.parse import quote, urlencode

import requests

logger = logging.getLogger(__name__)

BASE = "https://www.legislation.gov.uk"
SOURCE = "legislation.gov.uk"
USER_AGENT = "ilovelawyer-chat-wonder/1.0 (UK legislation lookup)"
# The first XML fetch of a section took ~6s from a cold start, so 30s rather than the MCP's 7s.
TIMEOUT_SECONDS = 30.0

_ATOM = "{http://www.w3.org/2005/Atom}"
_UKM = "{http://www.legislation.gov.uk/namespaces/metadata}"
_LEG = "{http://www.legislation.gov.uk/namespaces/legislation}"
_DC = "{http://purl.org/dc/elements/1.1/}"

_WAF_CHALLENGE = re.compile(rb"awswaf|AwsWafIntegration|aws-waf", re.IGNORECASE)
_ID_TYPE = re.compile(r"/id/([a-z]+)/")
_ID_YEAR_NUMBER = re.compile(r"/(\d{4})/(\d+)/?$")

# Elements that start/end a block of text; everything else (Addition, Substitution, Emphasis, ...) is inline.
_BLOCK = {
    "P1", "P2", "P3", "P4", "P5", "P6", "P7", "P1para", "P2para", "P3para", "P4para", "P5para", "P6para",
    "Pnumber", "Para", "Title", "ListItem", "OrderedList", "UnorderedList", "Text", "BlockAmendment",
}


class DirectLegislationError(Exception):
    """legislation.gov.uk could not be reached, refused, or returned something that is not legislation."""


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _fetch(url: str) -> bytes:
    """GET `url`, retrying once on a network error or a 5xx. A WAF challenge page is an error, not content."""
    for attempt in range(2):
        try:
            resp = requests.get(
                url,
                headers={"User-Agent": USER_AGENT, "Accept": "application/xml, application/atom+xml"},
                timeout=TIMEOUT_SECONDS,
            )
        except requests.RequestException as exc:
            if attempt == 0:
                time.sleep(0.4)
                continue
            raise DirectLegislationError(f"could not reach legislation.gov.uk: {exc}") from exc

        if resp.status_code == 404:
            raise DirectLegislationError(f"not found on legislation.gov.uk: {url}")
        if resp.status_code >= 500 and attempt == 0:
            time.sleep(0.4)
            continue
        try:
            resp.raise_for_status()
        except Exception as exc:
            raise DirectLegislationError(f"legislation.gov.uk returned an error for {url}: {exc}") from exc

        body = resp.content or b""
        if _WAF_CHALLENGE.search(body[:8000]):
            raise DirectLegislationError(f"legislation.gov.uk served an AWS WAF challenge page for {url}")
        return body
    raise DirectLegislationError(f"legislation.gov.uk did not answer for {url}")  # pragma: no cover


def _parse_xml(data: bytes) -> ET.Element:
    try:
        return ET.fromstring(data)
    except ET.ParseError as exc:
        raise DirectLegislationError(f"legislation.gov.uk response was not valid XML: {exc}") from exc


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_url(query: str, type: Optional[str], year: Optional[int], fulltext: bool) -> str:
    if fulltext:
        params = [("text", query)]
        if type:
            params.append(("type", str(type)))
        if year:
            params.append(("year", str(int(year))))
        return f"{BASE}/search/data.feed?{urlencode(params)}"
    path = [str(type) if type else "all"]
    if type and year:
        path.append(str(int(year)))
    return f"{BASE}/{'/'.join(path)}/data.feed?{urlencode([('title', query)])}"


def parse_feed(data: bytes, limit: int = 20, year: Optional[int] = None) -> List[Dict[str, Any]]:
    """Atom feed -> rows shaped like the MCP's legislation_search rows.

    Year and number come from the ukm:Year / ukm:Number metadata, not the entry id: older Acts are identified by
    regnal year (.../id/ukpga/Eliz2/1-2/20 is the Births and Deaths Registration Act 1953), but chat-wonder
    (uk_legal_mcp.urls) needs the calendar year to form .../ukpga/1953/20."""
    root = _parse_xml(data)
    rows: List[Dict[str, Any]] = []
    for entry in root.findall(f"{_ATOM}entry"):
        title = (entry.findtext(f"{_ATOM}title") or "").strip()
        entry_id = entry.findtext(f"{_ATOM}id") or ""
        type_match = _ID_TYPE.search(entry_id)
        if not title or not type_match:
            continue
        year_el, number_el = entry.find(f"{_UKM}Year"), entry.find(f"{_UKM}Number")
        row_year = year_el.get("Value") if year_el is not None else None
        row_number = number_el.get("Value") if number_el is not None else None
        if not (row_year and row_number):
            tail = _ID_YEAR_NUMBER.search(entry_id)
            if not tail:
                continue
            row_year, row_number = tail.group(1), tail.group(2)
        try:
            row_year, row_number = int(row_year), int(row_number)
        except ValueError:
            continue
        if year and row_year != int(year):
            continue
        doc_type = type_match.group(1)
        rows.append({
            "title": title,
            "type": doc_type,
            "year": row_year,
            "number": row_number,
            "url": f"{BASE}/{doc_type}/{row_year}/{row_number}",
        })
        if limit and len(rows) >= int(limit):
            break
    return rows


_TITLE_END = {"Act", "Order", "Regulations", "Rules", "Measure"}
_TITLE_CONNECTORS = {"of", "and", "the", "for", "to", "in", "on", "by", "with", "a"}
_MAX_TITLE_CANDIDATES = 4
_MAX_KEYWORDS = 6
_WORD = re.compile(r"[A-Za-z][A-Za-z'’-]*")
# Filler a client writes around the subject of a legal question; searching on it only narrows the match.
_STOPWORDS = frozenset(
    """that this with from have been were will would could should their there about which where while when what whom
    whose into onto upon over under than then them they your does done also such only other some more most very much many
    each both these those being having doing using used uses make made must might shall because between during before
    after since until through without within against among around whether however including include includes regarding
    concerning relating related person people different another every thing things want need needs help please explain
    tell""".split()
)


def _candidate_titles(query: str) -> List[str]:
    """Act/SI titles written inside a longer query, longest first: 'Births and Deaths Registration Act 1953
    correction error ...' -> ['Births and Deaths Registration Act 1953', 'Deaths Registration Act 1953', ...].

    legislation.gov.uk's title search matches only when the query is part of a title, so the model's habit of adding
    extra words after (or 'Under the' before) the title makes it return nothing."""
    tokens = query.split()
    candidates: List[str] = []
    for i, tok in enumerate(tokens):
        if tok.strip(".,;:()") not in _TITLE_END:
            continue
        end = i + 1 if i + 1 < len(tokens) and re.fullmatch(r"\d{4}", tokens[i + 1].strip(".,;:)")) else i
        start = i
        while start - 1 >= 0:
            prev = tokens[start - 1]
            if prev[:1].isupper() or prev.lower() in _TITLE_CONNECTORS:
                start -= 1
            else:
                break
        words = tokens[start:end + 1]
        for s in range(len(words)):
            if words[s].lower() in _TITLE_CONNECTORS:
                continue
            core = [w for w in words[s:] if w.strip(".,;:()") not in _TITLE_END
                    and not re.fullmatch(r"\d{4}", w.strip(".,;:)")) and w.lower() not in _TITLE_CONNECTORS]
            if not core:  # nothing but 'Act' and a year: would match every Act of that year
                continue
            phrase = " ".join(words[s:]).strip(".,;:")
            if phrase not in candidates:
                candidates.append(phrase)
    return candidates[:_MAX_TITLE_CANDIDATES]


def _keywords(query: str) -> List[str]:
    """The subject words of a sentence, in order, without filler — at most six."""
    seen: List[str] = []
    for word in _WORD.findall(query.lower()):
        if len(word) >= 4 and word not in _STOPWORDS and word not in seen:
            seen.append(word)
    return seen[:_MAX_KEYWORDS]


def search_legislation(query: str, type: Optional[str] = None, year: Optional[int] = None,
                       fulltext: bool = False, limit: int = 20) -> Dict[str, Any]:
    """Search legislation.gov.uk, widening step by step until something is found.

    Real queries from the model were a descriptive sentence, or an Act title plus extra words; a plain title search
    returned 0 for both and the answer went out with no law in it. So, stopping at the first step that has rows:
      1. the query as given (title search, or full text if `fulltext`);
      2. each Act/SI title found inside the query;
      3. the query's subject words joined with OR (full text), on primary legislation first unless the caller
         chose a type, then on everything.
    `strategy` in the result says which step answered."""
    query = (query or "").strip()
    steps: List[tuple] = []
    if fulltext:
        steps.append(("fulltext", search_url(query, type, year, True)))
    else:
        steps.append(("title", search_url(query, type, year, False)))
        for title in _candidate_titles(query):
            if title.lower() != query.lower():
                steps.append(("act-title", search_url(title, type, year, False)))
    keywords = _keywords(query)
    if keywords:
        joined = " OR ".join(keywords)
        if type is None:
            steps.append(("keywords", search_url(joined, "ukpga", year, True)))
        steps.append(("keywords", search_url(joined, type, year, True)))

    for strategy, url in steps:
        rows = parse_feed(_fetch(url), limit=limit, year=year)
        if rows:
            return {"results": rows, "total": len(rows), "source": SOURCE, "strategy": strategy}
    return {"results": [], "total": 0, "source": SOURCE, "strategy": "none"}


# ---------------------------------------------------------------------------
# One section
# ---------------------------------------------------------------------------

def _flatten(el: ET.Element) -> str:
    """Plain text of a provision in reading order: block elements are separated by spaces, inline ones
    (amendments such as <Addition>) run straight on, so 'section 29A</Addition>.' keeps its full stop."""
    parts: List[str] = []

    def walk(node: ET.Element) -> None:
        block = _local(node.tag) in _BLOCK
        if block:
            parts.append(" ")
        if node.text:
            parts.append(node.text)
        for child in node:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if block:
            parts.append(" ")

    walk(el)
    return re.sub(r"\s+", " ", "".join(parts)).strip()


def parse_section(data: bytes, type: str, year: Any, number: Any, section: str, max_chars: Optional[int] = 10000) -> Dict[str, Any]:
    root = _parse_xml(data)
    group = root.find(f".//{_LEG}P1group")
    if group is None:
        raise DirectLegislationError("no section text in the legislation.gov.uk response")

    title_el = group.find(f"{_LEG}Title")
    heading = _flatten(title_el) if title_el is not None else ""
    p1 = group.find(f"{_LEG}P1")
    body_el = p1 if p1 is not None else group
    number_el = p1.find(f"{_LEG}Pnumber") if p1 is not None else None
    section_number = _flatten(number_el) if number_el is not None else str(section)

    # Same layout the MCP returned: "<heading> <section no.> <1> <text> <2> <text> ..." — the body already
    # starts with the section number and the subsection numbers, so only the heading is put in front.
    content = f"{heading} {_flatten(body_el)}".strip()
    truncated = bool(max_chars) and len(content) > int(max_chars)
    if truncated:
        content = content[: int(max_chars)].rstrip()

    return {
        "title": heading,
        "section_number": section_number,
        "act_title": (root.findtext(f".//{_DC}title") or "").strip() or None,
        "content": content,
        "url": f"{BASE}/{type}/{int(year)}/{int(number)}/section/{section}",
        "truncated": truncated,
        "source": SOURCE,
    }


def get_section(type: str, year: Any, number: Any, section: str, max_chars: Optional[int] = 10000) -> Dict[str, Any]:
    url = f"{BASE}/{type}/{int(year)}/{int(number)}/section/{quote(str(section), safe='')}/data.xml"
    return parse_section(_fetch(url), type, year, number, section, max_chars)
