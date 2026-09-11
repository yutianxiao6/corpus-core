from __future__ import annotations

import unittest

from offline_rag.contracts.retrieval import SearchHit
from offline_rag.retrieval import normalized_weighted_fusion, reciprocal_rank_fusion


def hit(chunk_id: str, score: float) -> SearchHit:
    return SearchHit(chunk_id, score, {"chunk_id": chunk_id})


class FusionTests(unittest.TestCase):
    def test_rrf_deduplicates_and_rewards_both_origins(self) -> None:
        dense = [hit("dense-only", 0.99), hit("both", 0.8), hit("both", 0.7)]
        sparse = [hit("both", 12.0), hit("sparse-only", 10.0)]

        fused = reciprocal_rank_fusion(dense, sparse, constant=10)

        self.assertEqual(fused[0].chunk_id, "both")
        self.assertEqual(fused[0].origins, ("dense", "sparse"))
        self.assertEqual(len(fused), 3)
        self.assertAlmostEqual(fused[0].score, 1 / 12 + 1 / 11)

    def test_weighted_fusion_normalizes_incomparable_score_scales(self) -> None:
        dense = [hit("a", 0.9), hit("b", 0.1)]
        sparse = [hit("b", 100.0), hit("a", 10.0)]

        sparse_favored = normalized_weighted_fusion(
            dense, sparse, dense_weight=0.2, sparse_weight=0.8
        )
        dense_favored = normalized_weighted_fusion(
            dense, sparse, dense_weight=0.8, sparse_weight=0.2
        )

        self.assertEqual(sparse_favored[0].chunk_id, "b")
        self.assertEqual(dense_favored[0].chunk_id, "a")

    def test_equal_scores_and_invalid_weights_are_handled(self) -> None:
        fused = normalized_weighted_fusion([hit("b", 1), hit("a", 1)], [])
        self.assertEqual([item.chunk_id for item in fused], ["a", "b"])
        with self.assertRaises(ValueError):
            reciprocal_rank_fusion([], [], dense_weight=0, sparse_weight=0)


if __name__ == "__main__":
    unittest.main()
