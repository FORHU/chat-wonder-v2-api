# Legal Citation Observability

## Purpose

The legal response pipeline prefers citations that use juris.ph page URLs returned by
`search_jurisprudence`, `search_republic_acts`, `get_case`, and `get_republic_act`.

Legacy `/sources/<item_id>` links (from the removed local RAG) are rewritten to juris.ph
URLs from the active tool-result pool when possible.

In legal mode:
1. **Cite-gate** demotes markdown links whose `href` is not an exact URL from that pool
   (including truncated `juris.ph/case/...` placeholders). Link label is kept as plain text.
2. **Quote strip** demotes blockquote lines whose wording is not found in retrieved
   tool-result text (fabricated statute/case quotations).

Helpers live in `legal_citations.py`; `the_server.py` wraps them with metrics.

## Metrics

- `legal.citation_repair.count` — `/sources/` links rewritten to juris.ph URLs
- `legal.citation_gate.count` — unverified citation URLs demoted (label kept, href removed)
- `legal.citation_quote_strip.count` — unverified blockquotes removed

## Related

See ADR-0002 and `CONTEXT.md` (Legal search).
