"""Tests for bm25.BM25 — see bm25.py's module docstring for provenance and current (unwired)
status. Not tied to any server/session state, so these run in isolation."""

import unittest

from bm25 import BM25, rank, rrf_rank_scores


class Bm25ScoringTests(unittest.TestCase):
    def setUp(self):
        self.corpus = [
            "the claimant slipped on the wet floor near the loading bay",
            "the loading bay floor was wet because the drain had blocked",
            "judgment in 2019 UKSC 41 on the scope of the duty of care",
            "annual report prepared after the judgment was handed down",
        ]

    def test_exact_term_match_outranks_unrelated_document(self):
        bm25 = BM25(self.corpus)
        scores = bm25.scores("wet floor")
        self.assertGreater(scores[0], scores[2])
        self.assertGreater(scores[1], scores[2])

    def test_rare_term_outweighs_common_term(self):
        # "the" appears in every document (near-zero IDF); "judgment" appears in two.
        bm25 = BM25(self.corpus)
        scores = bm25.scores("judgment")
        self.assertGreater(scores[2], scores[0])
        self.assertGreater(scores[3], scores[0])

    def test_empty_query_scores_everything_zero(self):
        bm25 = BM25(self.corpus)
        self.assertEqual(bm25.scores(""), [0.0] * len(self.corpus))

    def test_query_term_absent_from_corpus_scores_zero(self):
        bm25 = BM25(self.corpus)
        self.assertEqual(bm25.scores("nonexistent gibberish zzqx"), [0.0] * len(self.corpus))

    def test_empty_corpus_returns_empty_scores(self):
        bm25 = BM25([])
        self.assertEqual(bm25.scores("anything"), [])

    def test_scores_normalised_peaks_at_one(self):
        bm25 = BM25(self.corpus)
        normalised = bm25.scores_normalised("wet floor")
        self.assertAlmostEqual(max(normalised), 1.0)
        for score in normalised:
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

    def test_scores_normalised_all_zero_when_no_match(self):
        bm25 = BM25(self.corpus)
        self.assertEqual(bm25.scores_normalised("zzqx nonexistent"), [0.0] * len(self.corpus))

    def test_longer_document_is_not_automatically_favoured(self):
        # Length normalization (b=0.75): a short doc mentioning the term once should not lose to
        # a much longer doc that only mentions it once amid unrelated padding.
        short = "duty of care applies here"
        long_padded = "duty of care applies here " + ("filler text " * 40)
        bm25 = BM25([short, long_padded])
        scores = bm25.scores("duty of care")
        self.assertGreaterEqual(scores[0], scores[1])


class Bm25RankTests(unittest.TestCase):
    def test_rank_orders_by_score_descending(self):
        corpus = ["cats and cats and birds", "dogs", "nothing relevant here at all"]
        order = rank(corpus, "dogs")
        self.assertEqual(order[0], 1)  # only doc containing "dogs" at all
        self.assertNotIn(1, order[1:])  # sanity: it isn't just returning input order

    def test_rank_respects_top_n(self):
        corpus = ["a b c", "a b", "a"]
        self.assertEqual(len(rank(corpus, "a b c", top_n=2)), 2)


class RrfRankScoresTests(unittest.TestCase):
    def test_item_ranked_well_in_both_lists_beats_item_ranked_well_in_only_one(self):
        # idx0: rank 1 in both lists. idx1: rank 1 embedding, absent from bm25. idx2: opposite.
        embedding_scores = [0.9, 0.8, 0.1]
        bm25_scores = [5.0, 0.0, 4.0]
        fused = rrf_rank_scores(embedding_scores, bm25_scores)
        self.assertGreater(fused[0], fused[1])
        self.assertGreater(fused[0], fused[2])

    def test_zero_or_negative_score_excludes_item_from_that_lists_ranking(self):
        # idx1 has a 0.0 bm25 score — must be excluded from the bm25 rank map entirely, not
        # merely ranked last (which would still contribute 1/(k+worst_rank)).
        embedding_scores = [0.5, 0.5]
        bm25_scores = [0.0, 0.0]
        fused = rrf_rank_scores(embedding_scores, bm25_scores)
        # Both tie on embedding rank (idx0 wins the tie-break by lower index) and both are
        # excluded from bm25 entirely, so idx0 > idx1, and neither is zero (both ranked on
        # the embedding list).
        self.assertGreater(fused[0], fused[1])
        self.assertGreater(fused[1], 0.0)

    def test_item_absent_from_both_lists_scores_zero(self):
        embedding_scores = [0.9, 0.0, 0.0]
        bm25_scores = [3.0, 0.0, 0.0]
        fused = rrf_rank_scores(embedding_scores, bm25_scores)
        self.assertEqual(fused[1], 0.0)
        self.assertEqual(fused[2], 0.0)

    def test_mismatched_list_lengths_sizes_to_the_longer_one(self):
        embedding_scores = [0.9, 0.8, 0.7, 0.6]
        bm25_scores = [5.0]
        fused = rrf_rank_scores(embedding_scores, bm25_scores)
        self.assertEqual(len(fused), 4)

    def test_peak_normalised_top_score_is_one(self):
        embedding_scores = [0.9, 0.5, 0.1]
        bm25_scores = [5.0, 2.0, 1.0]
        fused = rrf_rank_scores(embedding_scores, bm25_scores)
        self.assertEqual(max(fused), 1.0)

    def test_both_lists_empty_returns_empty(self):
        self.assertEqual(rrf_rank_scores([], []), [])

    def test_ties_break_by_lower_index(self):
        embedding_scores = [0.5, 0.5, 0.5]
        bm25_scores = [0.0, 0.0, 0.0]
        fused = rrf_rank_scores(embedding_scores, bm25_scores)
        self.assertGreater(fused[0], fused[1])
        self.assertGreater(fused[1], fused[2])

    def test_k_parameter_dampens_rank_one_dominance(self):
        # A larger k pulls every fused score closer together (before peak-normalization the
        # gap between rank 1 and rank 2 shrinks as k grows) — confirms k is actually wired
        # through rather than hardcoded.
        embedding_scores = [0.9, 0.8, 0.7]
        bm25_scores = [5.0, 4.0, 3.0]
        tight_k = rrf_rank_scores(embedding_scores, bm25_scores, k=1)
        loose_k = rrf_rank_scores(embedding_scores, bm25_scores, k=1000)
        self.assertGreater(tight_k[0] - tight_k[2], loose_k[0] - loose_k[2])


if __name__ == "__main__":
    unittest.main()
