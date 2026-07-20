"""Unit tests for legal fact-pattern boost and prefetch."""

import unittest
from unittest.mock import patch

from legal_fact_boost import apply_legal_fact_pattern_boost, prepare_legal_turn, prefetch_legal_authorities


class LegalFactBoostTests(unittest.TestCase):
    def test_opc_boost(self):
        out = apply_legal_fact_pattern_boost(
            "I am sole stockholder of an OPC and never designated a nominee."
        )
        self.assertIn("LEGAL_CHECKLIST", out)
        self.assertIn("Section 130", out)

    def test_cyber_boost(self):
        out = apply_legal_fact_pattern_boost(
            "Facebook post 18 months ago cyber libel and VAT for foreign customers"
        )
        self.assertIn("Causing", out)
        self.assertIn("RA 12023", out)
        self.assertIn("Article 33", out)

    def test_no_boost_unrelated(self):
        out = apply_legal_fact_pattern_boost("What is the weather today?")
        self.assertNotIn("LEGAL_CHECKLIST", out)

    def test_prefetch_free_patent(self):
        fake = {
            "success": True,
            "id": "ra11231",
            "title": "Agricultural Free Patent Reform Act",
            "url": "https://juris.ph/republic-act/ra11231",
            "type": "republic_act",
            "year": 2019,
            "document": {"summary": "Removes restrictions on agricultural free patents.", "year": 2019},
        }
        with patch("resources.functions.user_functions.get_republic_act", return_value=fake):
            entries, injection = prefetch_legal_authorities(
                "free patent land double sale half-brother naturalized American"
            )
        self.assertEqual(len(entries), 1)
        self.assertIn("11231", entries[0]["url"] + (entries[0].get("title") or ""))
        self.assertIn("PREFETCHED_AUTHORITIES", injection)
        self.assertIn("Agricultural Free Patent", injection)

    def test_prepare_legal_turn_combines(self):
        fake = {
            "success": True,
            "id": "x",
            "title": "RA 11232",
            "url": "https://juris.ph/republic-act/x",
            "document": {"summary": "Section 130 burden reversal for OPC."},
        }
        with patch("resources.functions.user_functions.get_republic_act", return_value=fake):
            text, entries = prepare_legal_turn("OPC sole stockholder nominee dispute")
        self.assertIn("LEGAL_CHECKLIST", text)
        self.assertIn("PREFETCHED_AUTHORITIES", text)
        self.assertTrue(entries)


if __name__ == "__main__":
    unittest.main()
