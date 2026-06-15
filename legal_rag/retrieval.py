import logging
import time
from typing import Any

from psycopg2.extras import RealDictCursor

from .db import LegalDatabase
from .embeddings import EmbeddingService

logger = logging.getLogger(__name__)


class HybridRetriever:
    def __init__(self, db: LegalDatabase, embeddings: EmbeddingService):
        self.db = db
        self.embeddings = embeddings

    def search(
        self,
        query: str,
        category=None,
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
            filters.append("d.category = %s")
            filter_params.append(category)
        if bucket_slug:
            filters.append("d.bucket_slug = %s")
            filter_params.append(bucket_slug)
        if year:
            filters.append("d.year = %s")
            filter_params.append(year)
        where_clause = f"AND {' AND '.join(filters)}" if filters else ""

        candidate_limit = max(limit * 4, 20)

        # Single round-trip: keyword and vector arms as CTEs, merged and ranked in one query.
        # snippet is chunk-level so we take MAX() — any chunk is fine as a preview.
        # full_text is only fetched when the caller needs it for LLM context (include_full_text=True),
        # avoiding large payload for plain conversational searches.
        full_text_inner = "d.full_text," if include_full_text else ""
        full_text_select = "MAX(full_text) AS full_text," if include_full_text else ""
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
                ORDER BY dc.embedding <=> %s::vector
                LIMIT %s
            ),
            merged AS (
                SELECT * FROM keyword
                UNION ALL
                SELECT * FROM vector
            )
            SELECT
                id, title, case_no, bucket_slug, category, year,
                source_url, s3_json_path, s3_manifest_path, summary,
                {full_text_select}
                COALESCE(
                    (ARRAY_AGG(snippet ORDER BY vector_score DESC NULLS LAST, keyword_score DESC NULLS LAST)
                     FILTER (WHERE vector_score > 0))[1],
                    MAX(snippet)
                ) AS snippet,
                MAX(keyword_score) AS keyword_score,
                MAX(vector_score) AS vector_score,
                MAX(keyword_score) * 0.45 + MAX(vector_score) * 0.55 AS final_score
            FROM merged
            GROUP BY id, title, case_no, bucket_slug, category, year,
                     source_url, s3_json_path, s3_manifest_path, summary
            ORDER BY final_score DESC
            LIMIT %s
        """

        keyword_params = [query, query, query, query, query, query, *filter_params, candidate_limit]
        vector_params = [vector_literal, *filter_params, vector_literal, candidate_limit]

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
