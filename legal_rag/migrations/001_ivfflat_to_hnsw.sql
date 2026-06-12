-- Migration: Replace IVFFlat vector index with HNSW
--
-- HNSW has lower query latency and better recall than IVFFlat at the cost of
-- more memory (~2–3x) and a slower index build. For read-heavy legal search
-- workloads this trade-off is always worth it.
--
-- HOW TO RUN:
--   This script MUST be run outside a transaction block.
--   Both CREATE INDEX CONCURRENTLY and DROP INDEX CONCURRENTLY are non-blocking —
--   they do not lock the table and queries continue during the build.
--
--   Option A — psql:
--     psql $DATABASE_URL -f 001_ivfflat_to_hnsw.sql
--
--   Option B — inside psql session:
--     \set AUTOCOMMIT on
--     \i 001_ivfflat_to_hnsw.sql
--
-- ESTIMATED TIME: depends on row count.
--   ~100k chunks  → a few minutes
--   ~1M chunks    → 15–30 minutes
--   ~10M chunks   → 1–2 hours
-- Queries keep working while the index builds.

-- Step 1: Build the HNSW index without locking the table.
-- m=16 (connections per layer) and ef_construction=64 are pgvector defaults —
-- a safe starting point. Increase ef_construction to 128 for higher recall
-- at the cost of a slower build.
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Step 2: Drop the old IVFFlat index once HNSW is ready.
-- CONCURRENTLY ensures no read/write lock on the table.
DROP INDEX CONCURRENTLY IF EXISTS idx_document_chunks_embedding_ivfflat;
