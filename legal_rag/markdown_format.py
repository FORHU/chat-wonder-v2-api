"""Convert legal document full_text into structured markdown for library display."""

from __future__ import annotations

import json
import logging
import re
from html import unescape

logger = logging.getLogger(__name__)

_MAX_SOURCE_CHARS = 48_000


def _strip_html(value: str) -> str:
    text = re.sub(r"<br\s*/?>", "\n", value, flags=re.I)
    text = re.sub(r"</p>", "\n\n", text, flags=re.I)
    text = re.sub(r"</div>", "\n", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _prepare_source_text(full_text: str) -> str:
    raw = (full_text or "").strip()
    if not raw:
        return ""
    if re.search(r"<[a-z][\s\S]*>", raw, flags=re.I):
        raw = _strip_html(raw)
    if len(raw) > _MAX_SOURCE_CHARS:
        raw = raw[:_MAX_SOURCE_CHARS] + "\n\n[... document truncated for formatting ...]"
    return raw


def _strip_title_dates(title: str) -> str:
    text = title.strip()
    months = (
        "January|February|March|April|May|June|July|August|September|October|November|December"
    )
    text = re.sub(rf",?\s*\b(?:{months})\s+\d{{1,2}},?\s*\d{{4}}\b", "", text, flags=re.I)
    text = re.sub(r",?\s*\b\d{4}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:150]


def _openai_client(openai_api_key: str, openai_base_url: str | None = None):
    from openai import OpenAI

    return (
        OpenAI(api_key=openai_api_key, base_url=openai_base_url)
        if openai_base_url
        else OpenAI(api_key=openai_api_key)
    )


def _parse_combined_json(raw: str) -> tuple[str, str]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Model returned invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("Model JSON must be an object")
    title = str(data.get("title") or "").strip()
    markdown = str(data.get("markdown") or "").strip()
    return title, markdown


def format_document_combined(
    full_text: str,
    *,
    existing_title: str | None = None,
    generate_title: bool = True,
    category: str | None = None,
    case_no: str | None = None,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    openai_base_url: str | None = None,
) -> tuple[str | None, str, bool]:
    """
    One OpenAI call: optional title + structured markdown.
    Returns (title, markdown, title_was_generated).
    """
    source = _prepare_source_text(full_text)
    if not source:
        return (existing_title, "", False)

    prior = (existing_title or "").strip()
    need_new_title = generate_title and not prior

    meta_lines = []
    if category:
        meta_lines.append(f"Category: {category}")
    if case_no:
        meta_lines.append(f"Case/number: {case_no}")
    meta_block = "\n".join(meta_lines)

    if prior:
        title_rules = (
            f'Use exactly this title unchanged in the JSON "title" field: "{prior}"'
        )
    elif need_new_title:
        title_rules = (
            'Generate "title": a short professional name (5-12 words) for this Philippine legal document. '
            "No quotes, dates, years, numbers, or trailing punctuation."
        )
    else:
        title_rules = 'Set "title" to an empty string.'

    prompt = f"""Analyze this Philippine legal document and return a single JSON object with exactly two keys:

{{
  "title": "...",
  "markdown": "..."
}}

Title rules:
- {title_rules}

Markdown rules:
- Preserve all legal meaning; do not summarize away articles, sections, or holdings.
- Use ## for major parts (e.g. Republic Act heading, syllabus, facts, ruling).
- Use ### for articles, sections, or numbered issues where appropriate.
- Use blockquotes (> ) for quoted statutory or judicial language.
- Use lists only when the source is clearly a list.
- Do not invent content.
- Do NOT put an H1 (#) title line in "markdown" — the title lives only in the "title" field.
- "markdown" must be valid Markdown only (no code fences wrapping the whole body).

{meta_block}

Document text:
{source}
"""

    client = _openai_client(openai_api_key, openai_base_url)
    completion = client.chat.completions.create(
        model=model,
        temperature=0.1,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You format Philippine legal documents. Always respond with valid JSON "
                    'containing "title" and "markdown" keys.'
                ),
            },
            {"role": "user", "content": prompt},
        ],
    )

    raw_title, markdown = _parse_combined_json(completion.choices[0].message.content or "")
    if prior:
        final_title = prior
        title_generated = False
    elif need_new_title and raw_title:
        final_title = _strip_title_dates(raw_title)
        title_generated = bool(final_title)
    else:
        final_title = _strip_title_dates(raw_title) if raw_title else None
        title_generated = False

    return (final_title or None, markdown, title_generated)


def format_document_to_markdown(
    full_text: str,
    *,
    title: str | None = None,
    category: str | None = None,
    case_no: str | None = None,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    openai_base_url: str | None = None,
) -> str:
    """Legacy helper — prefer format_document_combined."""
    _, markdown, _ = format_document_combined(
        full_text,
        existing_title=title,
        generate_title=False,
        category=category,
        case_no=case_no,
        openai_api_key=openai_api_key,
        model=model,
        openai_base_url=openai_base_url,
    )
    return markdown


def generate_document_title(
    source_text: str,
    *,
    openai_api_key: str,
    model: str = "gpt-4o-mini",
    openai_base_url: str | None = None,
) -> str:
    """Legacy helper — prefer format_document_combined."""
    title, _, generated = format_document_combined(
        source_text,
        generate_title=True,
        openai_api_key=openai_api_key,
        model=model,
        openai_base_url=openai_base_url,
    )
    return title if generated else ""


def prepend_title_heading(markdown: str, title: str | None) -> str:
    """Ensure formatted markdown opens with an H1 title line."""
    body = (markdown or "").strip()
    clean_title = (title or "").strip()
    if not clean_title:
        return body
    if body.startswith(f"# {clean_title}") or body.startswith(f"# {clean_title}\n"):
        return body
    return f"# {clean_title}\n\n{body}"
