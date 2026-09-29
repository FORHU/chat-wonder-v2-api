"""The case's Damages & Remedies model (request.case_damages from ilovelawyer-api) as a legal-prompt
block. Offline: no model call — the block's content and its tolerance of bad payloads are what the
legal personas depend on."""

import unittest

import the_server as srv

MODEL = {
    "currency": "PHP",
    "total": 918600,
    "low": 640000,
    "high": 1240000,
    "asOf": "2026-09-28",
    "provisional": True,
    "pendingEvidence": ["payroll certification"],
    "heads": [
        {"category": "ACTUAL", "label": "Actual (backwages)", "amount": 486000, "status": "PROVISIONAL",
         "basis": "27000 × 18 months", "pendingEvidence": "payroll certification"},
        {"category": "MORAL", "label": None, "amount": 200000, "status": "SUPPORTED", "basis": None},
        {"category": "ATTORNEYS_FEES", "label": "Attorney's fees", "amount": 78600.5, "status": "PROVISIONAL",
         "basis": "10% of ACTUAL + MORAL + EXEMPLARY"},
    ],
}


class CaseDamagesInjectionTests(unittest.TestCase):
    def test_lists_each_head_with_its_figure_status_and_basis(self):
        block = srv._build_case_damages_injection(MODEL)
        self.assertIn("[CASE DAMAGES MODEL]", block)
        self.assertIn("- Actual (backwages) (ACTUAL): ₱486,000 [provisional; 27000 × 18 months; waiting on payroll certification]", block)
        self.assertIn("- MORAL: ₱200,000 [supported]", block)
        self.assertIn("₱78,600.5", block)

    def test_states_the_total_range_accrual_and_provisional_status(self):
        block = srv._build_case_damages_injection(MODEL)
        self.assertIn(
            "TOTAL: ₱918,600 (exposure range ₱640,000 to ₱1,240,000), backwages accrued to 2026-09-28"
            " — PROVISIONAL until the payroll certification arrives",
            block,
        )

    def test_tells_the_model_to_quote_not_recompute(self):
        block = srv._build_case_damages_injection(MODEL)
        self.assertIn("Do not recompute", block)
        self.assertIn("Never present a provisional figure as final", block)

    def test_omits_the_range_when_it_equals_the_total_and_uses_pounds(self):
        model = {**MODEL, "currency": "GBP", "low": 918600, "high": 918600, "asOf": None, "provisional": False}
        block = srv._build_case_damages_injection(model)
        self.assertIn("TOTAL: £918,600\n", block)
        self.assertNotIn("exposure range", block)
        self.assertNotIn("PROVISIONAL", block)

    def test_bad_or_empty_payloads_inject_nothing(self):
        for bad in (None, "x", [], {}, {"heads": []}, {"heads": "nope"}, {"heads": [1, "two"]}):
            self.assertEqual(srv._build_case_damages_injection(bad), "", bad)

    def test_a_head_with_no_amount_says_so(self):
        block = srv._build_case_damages_injection({**MODEL, "heads": [{"category": "OTHER", "amount": None}]})
        self.assertIn("- OTHER: not set", block)


class CaseDamagesRequestTests(unittest.TestCase):
    def test_chat_request_accepts_and_defaults_the_field(self):
        self.assertIsNone(srv.ChatRequest(user_input="hi").case_damages)
        self.assertEqual(srv.ChatRequest(user_input="hi", case_damages=MODEL).case_damages["total"], 918600)

    def test_new_sessions_start_without_a_model(self):
        self.assertIsNone(srv.ChatState().case_damages)


if __name__ == "__main__":
    unittest.main()
