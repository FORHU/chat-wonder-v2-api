"""Synthetic relevance scoring for UK legislation search results.

uk-legal-mcp.fly.dev's `legislation_search` rows sometimes come back with a
null/missing `score`. Downstream, legal_citations.select_related_cases ranks
by `r.get("score") or 0.0`, so an unscored row silently sorts to the bottom
instead of being ranked sensibly. fill_missing_scores backfills a score for
those rows, combining positional rank decay, an authority boost for primary
Acts vs Statutory Instruments, and query-keyword hit density.
"""

from __future__ import annotations

from typing import Any, Dict, List

_TYPE_KEYS = ("type", "doc_type")
_TEXT_KEYS = ("content", "text", "snippet", "provision_text")


def _first_str(row: Dict[str, Any], keys) -> str:
    meta = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    for key in keys:
        val = row.get(key) or meta.get(key)
        if val:
            return str(val)
    return ""


def _calculate_synthetic_score(row: Dict[str, Any], rank: int, query: str) -> float:
    base_score = max(0.90 - (rank * 0.05), 0.40)

    doc_type = _first_str(row, _TYPE_KEYS).lower()
    authority_boost = 0.0
    if "ukpga" in doc_type or "act" in doc_type:
        authority_boost = 0.08
    elif "uksi" in doc_type or "instrument" in doc_type:
        authority_boost = 0.04

    content = _first_str(row, _TEXT_KEYS).lower()
    query_terms = [t.lower() for t in query.split() if len(t) > 3 and t.isalnum()]
    if query_terms and content:
        matches = sum(1 for term in query_terms if term in content)
        keyword_boost = (matches / len(query_terms)) * 0.10
    else:
        keyword_boost = 0.0

    final_score = base_score + authority_boost + keyword_boost
    return round(min(max(final_score, 0.0), 1.0), 4)


def fill_missing_scores(results: List[Dict[str, Any]], query: str) -> List[Dict[str, Any]]:
    """Backfill `score`/`score_source` on rows the server left unscored, in place."""
    for rank, row in enumerate(results):
        if not isinstance(row, dict):
            continue
        if row.get("score") is None:
            row["score"] = _calculate_synthetic_score(row, rank, query)
            row["score_source"] = "uk_legal_mcp_synthetic"
    return results
