# Legal Citation Observability

## Purpose

The legal response pipeline prefers citations that use juris.ph page URLs returned by
`search_jurisprudence`, `search_republic_acts`, `get_case`, and `get_republic_act`.

Legacy `/sources/<item_id>` links (from the removed local RAG) are rewritten to juris.ph
URLs from the active tool-result pool when possible.

## Metric

- `legal.citation_repair.count` — `/sources/` links rewritten to juris.ph URLs

## Related

See ADR-0002 and `CONTEXT.md` (Legal search).
