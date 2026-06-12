"""
Run from the project root:
    python -m legal_rag.migrations.run_hnsw_migration
"""
import os
import sys

import psycopg2
from pgvector.psycopg2 import register_vector


def main():
    dsn = os.environ.get("LEGAL_DATABASE_URL") or os.environ.get("POSTGRES_URL") or os.environ.get("DATABASE_URL")
    if not dsn:
        sys.exit("ERROR: set LEGAL_DATABASE_URL in your environment")

    dsn = dsn.split("?schema=")[0]
    conn = psycopg2.connect(dsn)
    conn.autocommit = True  # must be set before register_vector opens a transaction
    register_vector(conn)

    with conn.cursor() as cur:
        print("Checking existing indexes...")
        cur.execute("""
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'document_chunks'
            AND indexname IN (
                'idx_document_chunks_embedding_ivfflat',
                'idx_document_chunks_embedding_hnsw'
            )
        """)
        existing = {row[0] for row in cur.fetchall()}
        print(f"  Found: {existing or 'none'}")

        if "idx_document_chunks_embedding_hnsw" in existing:
            print("HNSW index already exists — nothing to do.")
        else:
            print("Building HNSW index (this runs without locking the table)...")
            cur.execute("""
                CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_hnsw
                    ON document_chunks USING hnsw (embedding vector_cosine_ops)
                    WITH (m = 16, ef_construction = 64)
            """)
            print("HNSW index created.")

        if "idx_document_chunks_embedding_ivfflat" in existing:
            print("Dropping old IVFFlat index...")
            cur.execute("DROP INDEX CONCURRENTLY IF EXISTS idx_document_chunks_embedding_ivfflat")
            print("IVFFlat index dropped.")
        else:
            print("IVFFlat index not present — skipping drop.")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
