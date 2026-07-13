import logging
import time
from typing import Any, Union

from psycopg2.extras import RealDictCursor

from .db import LegalDatabase
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)

KEYWORD_WEIGHT = 0.45
VECTOR_WEIGHT = 0.55


def chunk_final_score(keyword_score: float, vector_score: float) -> float:
    return float(keyword_score or 0) * KEYWORD_WEIGHT + float(vector_score or 0) * VECTOR_WEIGHT


def select_best_chunk_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the highest-scoring chunk row for snippet selection (not lexicographic max text)."""
    if not rows:
        raise ValueError("rows must be non-empty")
    return max(
        rows,
        key=lambda r: (
            chunk_final_score(r.get("keyword_score", 0), r.get("vector_score", 0)),
            float(r.get("vector_score") or 0),
            float(r.get("keyword_score") or 0),
        ),
    )


def aggregate_merged_hits(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    """
    Collapse chunk-level keyword/vector hits into document-level results.

    Document scores use MAX(keyword) / MAX(vector) across hits; snippet comes from
    the single best-scoring chunk for that document.
    """
    by_doc: dict[Any, list[dict[str, Any]]] = {}
    for row in rows:
        by_doc.setdefault(row["id"], []).append(row)

    results: list[dict[str, Any]] = []
    for doc_rows in by_doc.values():
        best = select_best_chunk_row(doc_rows)
        keyword_score = max(float(r.get("keyword_score") or 0) for r in doc_rows)
        vector_score = max(float(r.get("vector_score") or 0) for r in doc_rows)
        result = {
            "id": best["id"],
            "title": best.get("title"),
            "case_no": best.get("case_no"),
            "bucket_slug": best.get("bucket_slug"),
            "category": best.get("category"),
            "year": best.get("year"),
            "source_url": best.get("source_url"),
            "s3_json_path": best.get("s3_json_path"),
            "s3_manifest_path": best.get("s3_manifest_path"),
            "summary": best.get("summary"),
            "snippet": best.get("snippet"),
            "keyword_score": keyword_score,
            "vector_score": vector_score,
            "final_score": chunk_final_score(keyword_score, vector_score),
        }
        if "full_text" in best:
            result["full_text"] = best.get("full_text")
        results.append(result)

    results.sort(key=lambda r: r["final_score"], reverse=True)
    return results[:limit]


class HybridRetriever:
    def __init__(self, db: LegalDatabase, embeddings: EmbeddingService):
        self.db = db
        self.embeddings = embeddings

    def search(
        self,
        query: str,
        category: Union[str, list, None] = None,
        bucket_slug=None,
        year=None,
        limit: int = 10,
        include_full_text: bool = False,
    ) -> list[dict[str, Any]]:
        t0 = time.perf_counter()
        query_embedding = self.embeddings.embed_texts([query])[0]
        t_embed = time.perf_counter()
        vector_literal = "[" + ",".join(str(x) for x in query_embedding) + "]"

        filters = []
        filter_params: list[Any] = []
        if category:
            if isinstance(category, list) and len(category) > 1:
                filters.append("d.category = ANY(%s)")
                filter_params.append(category)
            else:
                cat = category[0] if isinstance(category, list) else category
                filters.append("d.category = %s")
                filter_params.append(cat)
        if bucket_slug:
            filters.append("d.bucket_slug = %s")
            filter_params.append(bucket_slug)
        if year:
            filters.append("d.year = %s")
            filter_params.append(year)
        where_clause = f"AND {' AND '.join(filters)}" if filters else ""

        candidate_limit = max(limit * 4, 20)

        # Single round-trip: keyword and vector arms as CTEs, then pick the best-scoring
        # chunk per document for snippet (ROW_NUMBER), while aggregating doc-level scores.
        full_text_inner = "d.full_text," if include_full_text else ""
        full_text_best = "b.full_text," if include_full_text else ""
        full_text_ranked = "full_text," if include_full_text else ""
        sql = f"""
            WITH keyword AS (
                SELECT
                    d.id,
                    d.title,
                    d.case_no,
                    d.bucket_slug,
                    d.category,
                    d.year,
                    d.source_url,
                    d.s3_json_path,
                    d.s3_manifest_path,
                    d.summary,
                    {full_text_inner}
                    dc.chunk_text AS snippet,
                    COALESCE(
                        ts_rank_cd(
                            to_tsvector('english', COALESCE(d.title,'') || ' ' || COALESCE(d.case_no,'') || ' ' || COALESCE(d.summary,'')),
                            plainto_tsquery('english', %s)
                        ), 0
                    ) +
                    CASE WHEN d.title ILIKE ('%%' || %s || '%%') OR d.case_no ILIKE ('%%' || %s || '%%') THEN 0.4 ELSE 0 END
                    AS keyword_score,
                    0.0::float AS vector_score
                FROM documents d
                JOIN document_chunks dc ON dc.document_id = d.id
                WHERE (
                    to_tsvector('english', COALESCE(d.title,'') || ' ' || COALESCE(d.case_no,'') || ' ' || COALESCE(d.summary,''))
                    @@ plainto_tsquery('english', %s)
                    OR d.title ILIKE ('%%' || %s || '%%')
                    OR d.case_no ILIKE ('%%' || %s || '%%')
                ) {where_clause}
                ORDER BY keyword_score DESC
                LIMIT %s
            ),
            vector AS (
                SELECT * FROM (
                    SELECT
                        d.id,
                        d.title,
                        d.case_no,
                        d.bucket_slug,
                        d.category,
                        d.year,
                        d.source_url,
                        d.s3_json_path,
                        d.s3_manifest_path,
                        d.summary,
                        {full_text_inner}
                        dc.chunk_text AS snippet,
                        0.0::float AS keyword_score,
                        (1 - (dc.embedding <=> %s::vector)) AS vector_score
                    FROM documents d
                    JOIN document_chunks dc ON dc.document_id = d.id
                    WHERE dc.embedding IS NOT NULL {where_clause}
                ) _vec
                ORDER BY vector_score DESC
                LIMIT %s
            ),
            merged AS (
                SELECT * FROM keyword
                UNION ALL
                SELECT * FROM vector
            ),
            doc_scores AS (
                SELECT
                    id,
                    MAX(keyword_score) AS keyword_score,
                    MAX(vector_score) AS vector_score,
                    MAX(keyword_score) * {KEYWORD_WEIGHT} + MAX(vector_score) * {VECTOR_WEIGHT} AS final_score
                FROM merged
                GROUP BY id
            ),
            best_chunk AS (
                SELECT
                    id, title, case_no, bucket_slug, category, year,
                    source_url, s3_json_path, s3_manifest_path, summary,
                    {full_text_ranked}
                    snippet,
                    keyword_score,
                    vector_score,
                    ROW_NUMBER() OVER (
                        PARTITION BY id
                        ORDER BY
                            (keyword_score * {KEYWORD_WEIGHT} + vector_score * {VECTOR_WEIGHT}) DESC,
                            vector_score DESC,
                            keyword_score DESC
                    ) AS rn
                FROM merged
            )
            SELECT
                b.id, b.title, b.case_no, b.bucket_slug, b.category, b.year,
                b.source_url, b.s3_json_path, b.s3_manifest_path, b.summary,
                {full_text_best}
                b.snippet AS snippet,
                s.keyword_score,
                s.vector_score,
                s.final_score
            FROM best_chunk b
            JOIN doc_scores s ON s.id = b.id
            WHERE b.rn = 1
            ORDER BY s.final_score DESC
            LIMIT %s
        """

        keyword_params = [query, query, query, query, query, query, *filter_params, candidate_limit]
        # vector_literal appears once: for the inner similarity expression.
        # ORDER BY is on the computed vector_score column (not the raw operator) so
        # PostgreSQL cannot use the HNSW index and falls back to an exact sequential
        # scan — correct and fast for small corpora; avoids HNSW pre-filter blindness.
        vector_params = [vector_literal, *filter_params, candidate_limit]

        with self.db.connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(sql, [*keyword_params, *vector_params, limit])
                rows = [dict(row) for row in cur.fetchall()]
        t_db = time.perf_counter()

        logger.info(
            "legal_search query=%r category=%r results=%d | embed=%.0fms db=%.0fms total=%.0fms",
            query[:80],
            category,
            len(rows),
            (t_embed - t0) * 1000,
            (t_db - t_embed) * 1000,
            (t_db - t0) * 1000,
        )

        if not include_full_text:
            for r in rows:
                r["full_text"] = None

        return rows
