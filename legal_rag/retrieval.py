from typing import Any

from psycopg2.extras import RealDictCursor

from .db import LegalDatabase
from .embeddings import EmbeddingService


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
    ) -> list[dict[str, Any]]:
        query_embedding = self.embeddings.embed_texts([query])[0]
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

        # No full_text in the JOIN — pulled separately after dedup to avoid
        # fetching 6 KB × N chunk rows for the same document.
        # WHERE clause uses @@ so PostgreSQL uses the GIN FTS index instead of full table scan.
        keyword_sql = f"""
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
                dc.chunk_text AS snippet,
                COALESCE(
                    ts_rank_cd(
                        to_tsvector('english', COALESCE(d.title,'') || ' ' || COALESCE(d.case_no,'') || ' ' || COALESCE(d.summary,'')),
                        plainto_tsquery('english', %s)
                    ), 0
                ) +
                CASE WHEN d.title ILIKE ('%%' || %s || '%%') OR d.case_no ILIKE ('%%' || %s || '%%') THEN 0.4 ELSE 0 END
                AS keyword_score
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
        """
        vector_sql = f"""
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
                dc.chunk_text AS snippet,
                (1 - (dc.embedding <=> %s::vector)) AS vector_score
            FROM documents d
            JOIN document_chunks dc ON dc.document_id = d.id
            WHERE dc.embedding IS NOT NULL {where_clause}
            ORDER BY dc.embedding <=> %s::vector
            LIMIT %s
        """

        with self.db.connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(keyword_sql, [query, query, query, query, query, query, *filter_params, candidate_limit])
                keyword_rows = [dict(row) for row in cur.fetchall()]
                cur.execute(vector_sql, [vector_literal, *filter_params, vector_literal, candidate_limit])
                vector_rows = [dict(row) for row in cur.fetchall()]

        merged: dict[int, dict[str, Any]] = {}
        for row in keyword_rows:
            item = merged.setdefault(row["id"], {**row, "keyword_score": 0.0, "vector_score": 0.0})
            if row.get("keyword_score", 0) > item["keyword_score"]:
                item.update(row)
                item["keyword_score"] = float(row.get("keyword_score") or 0.0)
        for row in vector_rows:
            item = merged.setdefault(row["id"], {**row, "keyword_score": 0.0, "vector_score": 0.0})
            if row.get("vector_score", 0) > item["vector_score"]:
                item.update(row)
                item["vector_score"] = float(row.get("vector_score") or 0.0)

        results = []
        for item in merged.values():
            item["final_score"] = item["keyword_score"] * 0.45 + item["vector_score"] * 0.55
            results.append(item)
        results.sort(key=lambda x: x.get("final_score", 0), reverse=True)
        top = results[:limit]

        # Batch-fetch full_text for the top documents in one query
        if top:
            doc_ids = [r["id"] for r in top]
            with self.db.connect() as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute(
                        "SELECT id, full_text FROM documents WHERE id = ANY(%s)",
                        (doc_ids,),
                    )
                    ft_map = {row["id"]: row["full_text"] for row in cur.fetchall()}
            for r in top:
                r["full_text"] = ft_map.get(r["id"])

        return top
