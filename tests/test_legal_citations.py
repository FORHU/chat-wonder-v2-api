"""Unit tests for legal citation repair and cite-gate."""

import unittest

from legal_citations import (
    apply_legal_citation_pipeline,
    collect_tool_result_urls,
    gate_unverified_legal_urls,
    repair_legal_source_links,
    select_related_cases,
    strip_unverified_blockquotes,
)


POOL = [
    {
        "id": "abc",
        "url": "https://juris.ph/case/abc",
        "title": "Manalo",
    },
    {
        "id": "ra1",
        "url": "https://juris.ph/republic-act/ra1",
        "document": {"url": "https://juris.ph/republic-act/ra1"},
    },
]


class LegalCitationsTests(unittest.TestCase):
    def test_collect_urls(self):
        urls = collect_tool_result_urls(POOL)
        self.assertEqual(
            urls,
            ["https://juris.ph/case/abc", "https://juris.ph/republic-act/ra1"],
        )

    def test_repair_sources_path(self):
        text = "See [Case Jurisprudence](/sources/99)"
        out = repair_legal_source_links(text, POOL)
        self.assertIn("https://juris.ph/case/abc", out)
        self.assertNotIn("/sources/", out)

    def test_gate_keeps_verified_url(self):
        text = "[Manalo Jurisprudence](https://juris.ph/case/abc)"
        out = gate_unverified_legal_urls(text, POOL)
        self.assertEqual(out, text)

    def test_gate_demotes_phantom_juris_url(self):
        text = (
            "See [Fake Jurisprudence](https://juris.ph/case/phantom-id) "
            "and [Good Jurisprudence](https://juris.ph/case/abc)."
        )
        out = gate_unverified_legal_urls(text, POOL)
        self.assertIn("Fake Jurisprudence", out)
        self.assertNotIn("phantom-id", out)
        self.assertIn("[Good Jurisprudence](https://juris.ph/case/abc)", out)

    def test_gate_demotes_truncated_placeholder(self):
        text = "[Case Jurisprudence](https://juris.ph/case/...)"
        out = gate_unverified_legal_urls(text, POOL)
        self.assertEqual(out, "Case Jurisprudence")

    def test_gate_empty_pool_demotes_all_http_links(self):
        text = "[Case Jurisprudence](https://juris.ph/case/abc)"
        out = gate_unverified_legal_urls(text, [])
        self.assertEqual(out, "Case Jurisprudence")

    def test_pipeline_legal_mode(self):
        text = (
            "A [Bad Jurisprudence](https://juris.ph/case/nope) and "
            "[Manalo Jurisprudence](https://juris.ph/case/abc) and "
            "[Legacy](/sources/1)."
        )
        out = apply_legal_citation_pipeline(text, POOL, legal_mode=True)
        self.assertNotIn("nope", out)
        self.assertIn('class="legal-ref jurisprudence"', out)
        self.assertIn("https://juris.ph/case/abc", out)
        self.assertNotIn("/sources/", out)

    def test_pipeline_non_legal_skips_gate_and_format(self):
        text = "[X Jurisprudence](https://juris.ph/case/nope)"
        out = apply_legal_citation_pipeline(text, POOL, legal_mode=False)
        # Non-legal: no gate, no HTML wrap; phantom URL left as model wrote it
        self.assertEqual(out, text)

    def test_strip_unverified_blockquote(self):
        pool = [
            {
                "id": "abc",
                "url": "https://juris.ph/case/abc",
                "snippet": "The Court held that floating status beyond six months is constructive dismissal.",
            }
        ]
        text = (
            "> *\"The Court held that floating status beyond six months is constructive dismissal.\"*\n"
            "> *\"Children born during marriage are legitimate under Article 177.\"*\n"
        )
        out = strip_unverified_blockquotes(text, pool)
        self.assertIn("floating status beyond six months", out)
        self.assertIn("Unverified quotation removed", out)
        self.assertNotIn("Article 177", out)

    def test_pipeline_strips_fake_quote_in_legal_mode(self):
        pool = [{"id": "x", "url": "https://juris.ph/case/x", "snippet": "Adequate capitalization matters under Section 130."}]
        text = '> *"Totally fabricated statute language about nominees."*\n'
        out = apply_legal_citation_pipeline(text, pool, legal_mode=True)
        self.assertIn("Unverified quotation removed", out)


class SelectRelatedCasesTests(unittest.TestCase):
    def test_empty_pool(self):
        self.assertEqual(select_related_cases([]), [])
        self.assertEqual(select_related_cases(None), [])

    def test_ranks_by_score_when_unvetted(self):
        pool = [
            {"id": "a", "title": "Low score", "score": 0.2, "type": "jurisprudence"},
            {"id": "b", "title": "High score", "score": 0.9, "type": "jurisprudence"},
        ]
        out = select_related_cases(pool)
        self.assertEqual([r["title"] for r in out], ["High score", "Low score"])

    def test_vetted_get_case_entry_outranks_unvetted_search_row(self):
        pool = [
            {"id": "a", "title": "Unvetted, high score", "score": 0.99, "type": "jurisprudence"},
            {"id": "b", "title": "Vetted via get_case", "score": None, "document": {"summary": "..."}},
        ]
        out = select_related_cases(pool)
        self.assertEqual(out[0]["title"], "Vetted via get_case")
        self.assertTrue(out[0]["vetted"])
        self.assertFalse(out[1]["vetted"])

    def test_dedupes_by_identity_preferring_vetted_entry(self):
        pool = [
            {"id": "same", "title": "Raw search row", "score": 0.5, "type": "jurisprudence"},
            {"id": "same", "title": "Vetted get_case entry", "document": {"summary": "..."}},
        ]
        out = select_related_cases(pool)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["title"], "Vetted get_case entry")
        self.assertTrue(out[0]["vetted"])

    def test_dedupes_by_url_when_no_id(self):
        pool = [
            {"url": "https://juris.ph/case/x", "title": "First"},
            {"url": "https://juris.ph/case/x", "title": "Second"},
        ]
        out = select_related_cases(pool)
        self.assertEqual(len(out), 1)

    def test_respects_limit(self):
        pool = [{"id": str(i), "title": f"Case {i}", "score": i} for i in range(10)]
        out = select_related_cases(pool, limit=3)
        self.assertEqual(len(out), 3)
        self.assertEqual([r["title"] for r in out], ["Case 9", "Case 8", "Case 7"])

    def test_prefetched_entry_is_treated_as_vetted(self):
        pool = [
            {"id": "a", "title": "Unvetted", "score": 0.9},
            {"id": "b", "title": "Prefetched", "prefetched": True},
        ]
        out = select_related_cases(pool)
        self.assertEqual(out[0]["title"], "Prefetched")

    def test_skips_rows_without_identity(self):
        pool = [{"title": "No id or url"}]
        self.assertEqual(select_related_cases(pool), [])

    def test_output_shape(self):
        pool = [
            {
                "id": "gr123",
                "type": "jurisprudence",
                "title": "People v. Doe",
                "url": "https://juris.ph/case/gr123",
                "case_number": "G.R. No. 123",
                "year": 2020,
                "snippet": "Some snippet.",
                "score": 0.75,
            }
        ]
        out = select_related_cases(pool)
        self.assertEqual(
            out,
            [
                {
                    "type": "jurisprudence",
                    "title": "People v. Doe",
                    "url": "https://juris.ph/case/gr123",
                    "case_number": "G.R. No. 123",
                    "ra_number": None,
                    "year": 2020,
                    "snippet": "Some snippet.",
                    "relevance": 0.75,
                    "vetted": False,
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
