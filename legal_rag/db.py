import json
from contextlib import contextmanager
from typing import Any

import psycopg2
import psycopg2.pool
from psycopg2.extras import Json, RealDictCursor
from pgvector.psycopg2 import register_vector


SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    source_hash TEXT NOT NULL UNIQUE,
    content_hash TEXT,
    bucket_slug TEXT NOT NULL,
    category TEXT NOT NULL,
    subcategory TEXT,
    title TEXT,
    case_no TEXT,
    year INT,
    source_url TEXT,
    metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    summary TEXT,
    concise_summary TEXT,
    full_text TEXT,
    full_text_source TEXT,
    formatted_markdown TEXT,
    s3_json_path TEXT NOT NULL,
    s3_manifest_path TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_documents_bucket_slug ON documents(bucket_slug);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_year ON documents(year);
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE INDEX IF NOT EXISTS idx_documents_fts ON documents USING GIN(to_tsvector('english', COALESCE(title,'') || ' ' || COALESCE(case_no,'') || ' ' || COALESCE(summary,'')));
CREATE INDEX IF NOT EXISTS idx_documents_title_trgm ON documents USING GIN(title gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_documents_case_no_trgm ON documents USING GIN(case_no gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_chunks_text_trgm ON document_chunks USING GIN(chunk_text gin_trgm_ops);

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    chunk_text TEXT NOT NULL,
    char_count INT NOT NULL,
    embedding vector(1536),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON document_chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);

CREATE TABLE IF NOT EXISTS ingestion_runs (
    id BIGSERIAL PRIMARY KEY,
    source_type TEXT NOT NULL,
    source_ref TEXT NOT NULL,
    status TEXT NOT NULL,
    started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    stats_json JSONB NOT NULL DEFAULT '{}'::jsonb,
    error_json JSONB
);
"""


class LegalDatabase:
    def __init__(self, dsn: str):
        self.dsn = dsn.split("?schema=")[0] if "?schema=" in dsn else dsn
        self._pool = psycopg2.pool.ThreadedConnectionPool(2, 10, self.dsn)

    @contextmanager
    def connect(self):
        conn = self._pool.getconn()
        register_vector(conn)
        try:
            yield conn
        finally:
            self._pool.putconn(conn)

    def ensure_schema(self) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(SCHEMA_SQL)
                cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS content_hash TEXT")
                cur.execute("ALTER TABLE documents ADD COLUMN IF NOT EXISTS formatted_markdown TEXT")
            conn.commit()

    def set_formatted_markdown(self, document_id: int, markdown: str) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET formatted_markdown = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (markdown, document_id),
                )
            conn.commit()

    def set_document_title(self, document_id: int, title: str) -> None:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE documents
                    SET title = %s, updated_at = NOW()
                    WHERE id = %s
                    """,
                    (title, document_id),
                )
            conn.commit()

    def create_ingestion_run(self, source_type: str, source_ref: str) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO ingestion_runs(source_type, source_ref, status)
                    VALUES (%s, %s, 'running')
                    RETURNING id
                    """,
                    (source_type, source_ref),
                )
                run_id = cur.fetchone()[0]
            conn.commit()
        return run_id

    def complete_ingestion_run(self, run_id: int, status: str, stats: dict[str, Any], error=None):
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingestion_runs
                    SET status = %s,
                        completed_at = NOW(),
                        stats_json = %s,
                        error_json = %s
                    WHERE id = %s
                    """,
                    (status, Json(stats), Json(error) if error else None, run_id),
                )
            conn.commit()

    def get_document_hashes(self, source_hash: str):
        with self.connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT id, content_hash
                    FROM documents
                    WHERE source_hash = %s
                    """,
                    (source_hash,),
                )
                row = cur.fetchone()
                return dict(row) if row else None

    def upsert_document(self, doc: dict[str, Any]) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO documents(
                        source_hash, content_hash, bucket_slug, category, subcategory, title, case_no, year, source_url,
                        metadata_json, summary, concise_summary, full_text, full_text_source,
                        s3_json_path, s3_manifest_path, updated_at
                    )
                    VALUES (%(source_hash)s, %(content_hash)s, %(bucket_slug)s, %(category)s, %(subcategory)s, %(title)s, %(case_no)s, %(year)s, %(source_url)s,
                            %(metadata_json)s, %(summary)s, %(concise_summary)s, %(full_text)s, %(full_text_source)s,
                            %(s3_json_path)s, %(s3_manifest_path)s, NOW())
                    ON CONFLICT (source_hash) DO UPDATE SET
                        content_hash = EXCLUDED.content_hash,
                        bucket_slug = EXCLUDED.bucket_slug,
                        category = EXCLUDED.category,
                        subcategory = EXCLUDED.subcategory,
                        title = EXCLUDED.title,
                        case_no = EXCLUDED.case_no,
                        year = EXCLUDED.year,
                        source_url = EXCLUDED.source_url,
                        metadata_json = EXCLUDED.metadata_json,
                        summary = EXCLUDED.summary,
                        concise_summary = EXCLUDED.concise_summary,
                        full_text = EXCLUDED.full_text,
                        full_text_source = EXCLUDED.full_text_source,
                        s3_json_path = EXCLUDED.s3_json_path,
                        s3_manifest_path = EXCLUDED.s3_manifest_path,
                        updated_at = NOW()
                    RETURNING id
                    """,
                    {**doc, "metadata_json": Json(doc.get("metadata_json", {}))},
                )
                document_id = cur.fetchone()[0]
            conn.commit()
        return document_id

    def replace_document_chunks(self, document_id: int, chunks: list[str], embeddings: list[list[float]]) -> int:
        with self.connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM document_chunks WHERE document_id = %s", (document_id,))
                for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
                    vector_literal = "[" + ",".join(str(x) for x in embedding) + "]"
                    cur.execute(
                        """
                        INSERT INTO document_chunks(document_id, chunk_index, chunk_text, char_count, embedding)
                        VALUES (%s, %s, %s, %s, %s::vector)
                        """,
                        (document_id, idx, chunk, len(chunk), vector_literal),
                    )
            conn.commit()
        return len(chunks)

    def list_buckets(self) -> list[dict[str, Any]]:
        with self.connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(
                    """
                    SELECT category, bucket_slug, COUNT(*) AS document_count, MIN(year) AS min_year, MAX(year) AS max_year
                    FROM documents
                    GROUP BY category, bucket_slug
                    ORDER BY category, bucket_slug
                    """
                )
                return [dict(row) for row in cur.fetchall()]

    def get_document(self, doc_id: int):
        with self.connect() as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("SELECT * FROM documents WHERE id = %s", (doc_id,))
                row = cur.fetchone()
                return dict(row) if row else None

