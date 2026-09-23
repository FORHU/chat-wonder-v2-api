"""Unit tests for ph_legal_query.plan_ra_query (FORHU/chat-wonder-v2-api#75)."""

import unittest

from ph_legal_query import document_number, plan_ra_query


class PlanRaQueryTests(unittest.TestCase):
    def test_ordinary_query_untouched(self):
        plan = plan_ra_query("violence against women")
        self.assertEqual(plan.kind, "search")
        self.assertEqual(plan.queries, ["violence against women"])
        self.assertIsNone(plan.ra_number)

    def test_expands_acronym_and_never_sends_it_upstream(self):
        plan = plan_ra_query("VAWC")
        self.assertEqual(plan.kind, "search")
        self.assertEqual(plan.ra_number, "9262")
        self.assertIn("Republic Act No. 9262", plan.queries)
        self.assertIn("R.A. No. 9262", plan.queries)
        self.assertIn("RA 9262", plan.queries)
        self.assertIn("Anti-Violence Against Women and Their Children Act of 2004", plan.queries)
        self.assertNotIn("VAWC", plan.queries)

    def test_acronym_matching_ignores_case_dots_and_spaces(self):
        for q in ["vawc", "V.A.W.C.", " Vawc "]:
            self.assertEqual(plan_ra_query(q).ra_number, "9262", q)
        self.assertEqual(plan_ra_query("RH Law").ra_number, "10354")

    def test_ipra_maps_to_8371_not_intellectual_property(self):
        self.assertEqual(plan_ra_query("IPRA").ra_number, "8371")

    def test_every_ra_number_format_normalizes_to_the_same_digits(self):
        forms = [
            "RA 9262", "R.A. 9262", "RA No. 9262", "R.A. No. 9262", "RA No.9262", "RA #9262",
            "RA# 9262", "RA9262", "Republic Act 9262", "Republic Act No. 9262",
            "Republic Act No.9262", "Republic Act #9262", "Republic Act Number 9262",
            "republic act no 9262", "9262", "No. 9262", "#9262",
        ]
        for q in forms:
            plan = plan_ra_query(q)
            self.assertEqual(plan.kind, "search", q)
            self.assertEqual(plan.ra_number, "9262", q)
            self.assertEqual(plan.queries, ["Republic Act No. 9262", "R.A. No. 9262", "RA 9262"], q)

    def test_does_not_treat_a_short_bare_number_as_a_law(self):
        plan = plan_ra_query("12")
        self.assertEqual(plan.kind, "search")
        self.assertEqual(plan.queries, ["12"])
        self.assertIsNone(plan.ra_number)

    def test_only_expands_a_whole_query_acronym(self):
        plan = plan_ra_query("vawc protection order")
        self.assertEqual(plan.queries, ["vawc protection order"])
        self.assertIsNone(plan.ra_number)

    def test_issuances_juris_ph_does_not_index(self):
        cases = [
            ("EO 209", "Executive Order No. 209"),
            ("E.O. No. 292", "Executive Order No. 292"),
            ("PD 442", "Presidential Decree No. 442"),
            ("P.D. No. 1529", "Presidential Decree No. 1529"),
            ("AO 25", "Administrative Order No. 25"),
            ("MO 32", "Memorandum Order No. 32"),
            ("Memorandum Circular 8", "Memorandum Circular No. 8"),
            ("BP 22", "Batas Pambansa Blg. 22"),
            ("B.P. Blg. 22", "Batas Pambansa Blg. 22"),
            ("CA 141", "Commonwealth Act No. 141"),
            ("Act No. 3815", "Act No. 3815"),
        ]
        for q, label in cases:
            plan = plan_ra_query(q)
            self.assertEqual(plan.kind, "unindexed", q)
            self.assertEqual(plan.label, label, q)

    def test_pd_442_is_never_read_as_ra_442(self):
        self.assertEqual(plan_ra_query("PD 442").kind, "unindexed")

    def test_still_searches_by_name(self):
        self.assertEqual(plan_ra_query("Labor Code").kind, "search")

    def test_empty_query(self):
        plan = plan_ra_query("")
        self.assertEqual(plan.kind, "search")
        self.assertEqual(plan.queries, [""])


class DocumentNumberTests(unittest.TestCase):
    def test_first_run_of_digits(self):
        self.assertEqual(document_number("RA 9262"), "9262")
        self.assertEqual(document_number("No. 8371"), "8371")
        self.assertEqual(document_number("R.A. No. 9208 (as amended by R.A. No. 10364)"), "9208")
        self.assertEqual(document_number(None), "")


if __name__ == "__main__":
    unittest.main()
