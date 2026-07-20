"""Unit tests for dynamic legal fact-pattern boost and prefetch."""

import unittest
from unittest.mock import patch

from legal_fact_boost import (
    apply_legal_fact_pattern_boost,
    prepare_legal_turn,
    prefetch_legal_authorities,
)


class LegalFactBoostTests(unittest.TestCase):
    def test_every_question_gets_general_protocol(self):
        out = apply_legal_fact_pattern_boost(
            "What are the elements of estafa under the Revised Penal Code?"
        )
        self.assertIn("LEGAL_ANALYSIS_PROTOCOL", out)
        self.assertIn("Issue-spot", out)
        self.assertIn("AnyCase-style", out)

    def test_opc_hint_still_fires(self):
        out = apply_legal_fact_pattern_boost(
            "I am sole stockholder of an OPC and never designated a nominee."
        )
        self.assertIn("LEGAL_ANALYSIS_PROTOCOL", out)
        self.assertIn("DOCTRINE_HINTS", out)
        self.assertIn("Sec. 130", out)

    def test_cyber_hint(self):
        out = apply_legal_fact_pattern_boost(
            "Facebook post 18 months ago cyber libel and VAT for foreign customers"
        )
        self.assertIn("Causing", out)
        self.assertIn("RA 12023", out)
        self.assertIn("Art. 33", out)

    def test_divorce_hint_names_manalo_line(self):
        out = apply_legal_fact_pattern_boost(
            "Japanese divorce, I filed, can I remarry in the Philippines?"
        )
        self.assertIn("Manalo", out)
        self.assertIn("Ordaneza", out)
        self.assertIn("annotation", out)

    def test_unrelated_still_gets_protocol_not_only_hints(self):
        out = apply_legal_fact_pattern_boost("What is the weather today?")
        self.assertIn("LEGAL_ANALYSIS_PROTOCOL", out)

    def test_prefetch_explicit_ra_number(self):
        fake = {
            "success": True,
            "id": "ra11313",
            "title": "Safe Spaces Act",
            "url": "https://juris.ph/republic-act/ra11313",
            "type": "republic_act",
            "document": {"summary": "Defines gender-based streets and public spaces harassment.", "year": 2019},
        }
        with patch("resources.functions.user_functions.get_republic_act", return_value=fake):
            entries, injection = prefetch_legal_authorities(
                "Please explain RA 11313 and how it applies to workplace harassment."
            )
        self.assertEqual(len(entries), 1)
        self.assertIn("11313", entries[0]["url"] + entries[0]["title"])
        self.assertIn("PREFETCHED_AUTHORITIES", injection)

    def test_prefetch_keyword_opc(self):
        fake = {
            "success": True,
            "id": "x",
            "title": "Revised Corporation Code",
            "url": "https://juris.ph/republic-act/x",
            "document": {"summary": "Section 130 burden reversal for OPC."},
        }
        with patch("resources.functions.user_functions.get_republic_act", return_value=fake):
            text, entries = prepare_legal_turn("OPC sole stockholder nominee dispute")
        self.assertIn("LEGAL_ANALYSIS_PROTOCOL", text)
        self.assertTrue(entries)


if __name__ == "__main__":
    unittest.main()
