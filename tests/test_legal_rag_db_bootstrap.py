"""Unit tests for legal RAG DB schema bootstrap ordering (no live Postgres required)."""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from legal_rag.db import SCHEMA_SQL, LegalDatabase  # noqa: E402


class TestSchemaSqlOrdering(unittest.TestCase):
    def test_document_chunks_table_before_chunk_indexes(self):
        sql = re.sub(r"\s+", " ", SCHEMA_SQL)
        table_pos = sql.upper().find("CREATE TABLE IF NOT EXISTS DOCUMENT_CHUNKS")
        self.assertGreaterEqual(table_pos, 0, "document_chunks table must be declared")

        for index_name in (
            "idx_chunks_text_trgm",
            "idx_document_chunks_document_id",
            "idx_document_chunks_embedding_hnsw",
        ):
            idx_pos = sql.lower().find(index_name.lower())
            self.assertGreaterEqual(idx_pos, 0, f"missing index {index_name}")
            self.assertLess(
                table_pos,
                idx_pos,
                f"{index_name} must appear after CREATE TABLE document_chunks",
            )

    def test_vector_extension_before_document_chunks_table(self):
        sql = re.sub(r"\s+", " ", SCHEMA_SQL).upper()
        ext_pos = sql.find("CREATE EXTENSION IF NOT EXISTS VECTOR")
        table_pos = sql.find("CREATE TABLE IF NOT EXISTS DOCUMENT_CHUNKS")
        self.assertGreaterEqual(ext_pos, 0)
        self.assertLess(ext_pos, table_pos)


class TestEnsureSchemaBootstrap(unittest.TestCase):
    def test_ensure_schema_disables_pgvector_registration(self):
        db = LegalDatabase.__new__(LegalDatabase)
        conn = MagicMock()
        cur = MagicMock()
        conn.cursor.return_value.__enter__.return_value = cur
        conn.cursor.return_value.__exit__.return_value = False

        with patch.object(LegalDatabase, "connect") as connect_mock:
            connect_mock.return_value.__enter__.return_value = conn
            connect_mock.return_value.__exit__.return_value = False
            db.ensure_schema()

        connect_mock.assert_called_once_with(register_pgvector=False)
        # SCHEMA_SQL is executed on the raw connection.
        executed = [call.args[0] for call in cur.execute.call_args_list if call.args]
        self.assertTrue(any("CREATE EXTENSION IF NOT EXISTS vector" in sql for sql in executed))

    def test_connect_skips_register_vector_when_disabled(self):
        db = LegalDatabase.__new__(LegalDatabase)
        db._pool = MagicMock()
        fake_conn = object()
        db._pool.getconn.return_value = fake_conn

        with patch("legal_rag.db.register_vector") as register_mock:
            with db.connect(register_pgvector=False) as conn:
                self.assertIs(conn, fake_conn)
            register_mock.assert_not_called()
            db._pool.putconn.assert_called_once_with(fake_conn)

    def test_connect_registers_vector_by_default(self):
        db = LegalDatabase.__new__(LegalDatabase)
        db._pool = MagicMock()
        fake_conn = object()
        db._pool.getconn.return_value = fake_conn

        with patch("legal_rag.db.register_vector") as register_mock:
            with db.connect() as conn:
                self.assertIs(conn, fake_conn)
            register_mock.assert_called_once_with(fake_conn)


if __name__ == "__main__":
    unittest.main()
