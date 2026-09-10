"""Tests for uk_legal_mcp.scoring.fill_missing_scores."""

import unittest

from uk_legal_mcp.scoring import fill_missing_scores


class FillMissingScoresTests(unittest.TestCase):
    def test_rank_decay_orders_unscored_rows(self):
        rows = [{"title": "first"}, {"title": "second"}, {"title": "third"}]
        fill_missing_scores(rows, "insolvency")
        self.assertGreater(rows[0]["score"], rows[1]["score"])
        self.assertGreater(rows[1]["score"], rows[2]["score"])

    def test_authority_boost_ranks_act_above_instrument_above_unknown(self):
        # Score each alone at rank 0 so positional decay can't confound the comparison.
        act = fill_missing_scores([{"title": "act", "type": "ukpga"}], "companies")[0]
        si = fill_missing_scores([{"title": "si", "type": "uksi"}], "companies")[0]
        unknown = fill_missing_scores([{"title": "unknown", "type": "other"}], "companies")[0]
        self.assertGreater(act["score"], si["score"])
        self.assertGreater(si["score"], unknown["score"])

    def test_keyword_density_boosts_matching_row(self):
        rows = [
            {"title": "no match", "content": "unrelated text about cats"},
            {"title": "match", "content": "duties of company directors during insolvency"},
        ]
        fill_missing_scores(rows, "director insolvency duties")
        self.assertGreater(rows[1]["score"], rows[0]["score"])

    def test_existing_score_is_left_untouched(self):
        rows = [{"title": "already scored", "score": 0.73}]
        fill_missing_scores(rows, "anything")
        self.assertEqual(rows[0]["score"], 0.73)
        self.assertNotIn("score_source", rows[0])

    def test_synthesized_rows_are_marked_and_clamped(self):
        rows = [{"title": "row", "type": "ukpga", "content": "director duties insolvency"}]
        fill_missing_scores(rows, "director duties insolvency")
        self.assertEqual(rows[0]["score_source"], "uk_legal_mcp_synthetic")
        self.assertGreaterEqual(rows[0]["score"], 0.0)
        self.assertLessEqual(rows[0]["score"], 1.0)


if __name__ == "__main__":
    unittest.main()
