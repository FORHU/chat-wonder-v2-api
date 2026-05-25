#!/usr/bin/env python3
"""
Batch-generate formatted_markdown for legal RAG documents.

Run on EC2 (or locally) where LEGAL_DATABASE_URL and OPENAI_API_KEY are set:

  python scripts/format_legal_markdown.py --limit 100
  python scripts/format_legal_markdown.py --all          # every unformatted doc (~43k; long run)
  python scripts/format_legal_markdown.py --id 150
  python scripts/format_legal_markdown.py --force --all  # reformat entire corpus

Uses the same formatter as POST /api/legal/format-document/{id}.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

# Project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()
user_env = os.path.join("resources", "functions", "user_functions.env")
if os.path.exists(user_env):
    load_dotenv(user_env, override=True)

from legal_rag.db import LegalDatabase
from legal_rag.markdown_format import format_document_combined, prepend_title_heading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Format legal documents to markdown")
    parser.add_argument("--limit", type=int, default=50, help="Max documents per run (ignored with --all)")
    parser.add_argument("--all", action="store_true", help="Process every matching document (no LIMIT)")
    parser.add_argument("--id", type=int, help="Format a single document id")
    parser.add_argument("--force", action="store_true", help="Reformat even if formatted_markdown exists")
    parser.add_argument("--delay", type=float, default=0.5, help="Seconds between OpenAI calls")
    parser.add_argument(
        "--no-title",
        action="store_true",
        help="Skip AI title generation for documents with empty title",
    )
    args = parser.parse_args()

    db_url = os.getenv("LEGAL_DATABASE_URL", "")
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not db_url:
        logger.error("LEGAL_DATABASE_URL is not set")
        sys.exit(1)
    if not api_key:
        logger.error("OPENAI_API_KEY is not set")
        sys.exit(1)

    db = LegalDatabase(db_url)
    db.ensure_schema()

    if args.id:
        doc_ids = [args.id]
    else:
        limit_clause = "" if args.all else "LIMIT %s"
        with db.connect() as conn:
            with conn.cursor() as cur:
                if args.force:
                    sql = f"""
                        SELECT id FROM documents
                        WHERE full_text IS NOT NULL AND length(trim(full_text)) > 100
                        ORDER BY id
                        {limit_clause}
                    """
                    cur.execute(sql, () if args.all else (args.limit,))
                else:
                    sql = f"""
                        SELECT id FROM documents
                        WHERE formatted_markdown IS NULL
                          AND full_text IS NOT NULL
                          AND length(trim(full_text)) > 100
                        ORDER BY id
                        {limit_clause}
                    """
                    cur.execute(sql, () if args.all else (args.limit,))
                doc_ids = [row[0] for row in cur.fetchall()]

    if not doc_ids:
        logger.info("No documents to format.")
        return

    logger.info("Formatting %s document(s)...", len(doc_ids))
    model = os.getenv("LEGAL_CHAT_MODEL", "gpt-4o-mini")
    ok = 0
    failed = 0
    titles_generated = 0

    for doc_id in doc_ids:
        doc = db.get_document(doc_id)
        if not doc:
            logger.warning("[%s] not found", doc_id)
            failed += 1
            continue

        existing_title = (doc.get("title") or "").strip() or None
        if (doc.get("formatted_markdown") or "").strip() and not args.force:
            if existing_title or args.no_title:
                logger.info("[%s] already formatted — skip", doc_id)
                ok += 1
                continue

        source = doc.get("full_text") or doc.get("summary") or doc.get("concise_summary") or ""
        try:
            title, markdown, title_generated = format_document_combined(
                str(source),
                existing_title=existing_title,
                generate_title=not args.no_title,
                category=doc.get("category"),
                case_no=doc.get("case_no"),
                openai_api_key=api_key,
                model=model,
                openai_base_url=os.getenv("OPENAI_BASE_URL"),
            )
            if not markdown:
                logger.warning("[%s] empty markdown", doc_id)
                failed += 1
                continue
            if title_generated and title:
                db.set_document_title(doc_id, title)
                titles_generated += 1
                logger.info("[%s] title: %s", doc_id, title)
            markdown = prepend_title_heading(markdown, title or existing_title)
            db.set_formatted_markdown(doc_id, markdown)
            logger.info("[%s] saved (%s chars)", doc_id, len(markdown))
            ok += 1
        except Exception as exc:
            logger.error("[%s] failed: %s", doc_id, exc)
            failed += 1

        if args.delay > 0:
            time.sleep(args.delay)

    logger.info("Done. ok=%s failed=%s titles_generated=%s", ok, failed, titles_generated)


if __name__ == "__main__":
    main()
