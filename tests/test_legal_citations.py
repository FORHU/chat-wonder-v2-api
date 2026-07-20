"""Unit tests for legal citation repair and cite-gate."""

import unittest

from legal_citations import (
    apply_legal_citation_pipeline,
    collect_tool_result_urls,
    gate_unverified_legal_urls,
    repair_legal_source_links,
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


if __name__ == "__main__":
    unittest.main()
