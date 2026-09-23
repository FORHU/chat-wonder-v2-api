"""Unit tests for uk_legal_query (FORHU/chat-wonder-v2-api#94)."""

import unittest

from uk_legal_query import DocRef, plan_uk_legislation_query, prefer_exact_ref


class PlanUkLegislationQueryTests(unittest.TestCase):
    def test_ordinary_query_untouched(self):
        plan = plan_uk_legislation_query("data protection obligations for schools")
        self.assertEqual(plan.kind, "plain")
        self.assertEqual(plan.query, "data protection obligations for schools")
        self.assertIsNone(plan.ref)

    def test_expands_hra_to_its_full_title(self):
        plan = plan_uk_legislation_query("HRA")
        self.assertEqual(plan.kind, "alias")
        self.assertEqual(plan.query, "Human Rights Act 1998")
        self.assertEqual(plan.ref, DocRef("ukpga", 1998, 42))

    def test_matching_ignores_case_and_punctuation(self):
        for q in ["hra", "H.R.A.", " Hra "]:
            self.assertEqual(plan_uk_legislation_query(q).query, "Human Rights Act 1998", q)

    def test_every_verified_alias_resolves(self):
        expected = {
            "PACE": "Police and Criminal Evidence Act 1984",
            "FOIA": "Freedom of Information Act 2000",
            "DPA": "Data Protection Act 2018",
            "TUPE": "The Transfer of Undertakings (Protection of Employment) Regulations 2006",
            "IHTA": "Inheritance Tax Act 1984",
            "MHA": "Mental Health Act 1983",
            "EA": "Equality Act 2010",
        }
        for acronym, title in expected.items():
            plan = plan_uk_legislation_query(acronym)
            self.assertEqual(plan.kind, "alias", acronym)
            self.assertEqual(plan.query, title, acronym)
            self.assertIsNotNone(plan.ref, acronym)

    def test_only_expands_a_whole_query_acronym(self):
        plan = plan_uk_legislation_query("HRA damages claim")
        self.assertEqual(plan.kind, "plain")
        self.assertEqual(plan.query, "HRA damages claim")

    def test_empty_query(self):
        plan = plan_uk_legislation_query("")
        self.assertEqual(plan.kind, "plain")
        self.assertEqual(plan.query, "")


class PreferExactRefTests(unittest.TestCase):
    def test_moves_the_exact_document_to_the_front(self):
        ref = DocRef("ukpga", 2010, 15)
        rows = [
            {"title": "Worker Protection (Amendment of Equality Act 2010) Act 2023", "type": "ukpga", "year": 2023, "number": 51},
            {"title": "Equality Act 2010", "type": "ukpga", "year": 2010, "number": 15},
        ]
        out = prefer_exact_ref(rows, ref)
        self.assertEqual(out[0]["title"], "Equality Act 2010")
        self.assertEqual(len(out), 2)

    def test_leaves_the_order_unchanged_when_the_exact_document_is_absent(self):
        ref = DocRef("ukpga", 2010, 15)
        rows = [{"title": "Something else", "type": "ukpga", "year": 1999, "number": 1}]
        self.assertEqual(prefer_exact_ref(rows, ref), rows)

    def test_leaves_the_order_unchanged_when_it_is_already_first(self):
        ref = DocRef("ukpga", 1998, 42)
        rows = [{"title": "Human Rights Act 1998", "type": "ukpga", "year": 1998, "number": 42}]
        self.assertEqual(prefer_exact_ref(rows, ref), rows)

    def test_handles_an_empty_row_list(self):
        self.assertEqual(prefer_exact_ref([], DocRef("ukpga", 1998, 42)), [])


if __name__ == "__main__":
    unittest.main()
