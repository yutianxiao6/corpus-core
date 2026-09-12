from __future__ import annotations

import math
import unittest

from corpuscore.embeddings import HashedLexicalSparseEmbedding
from corpuscore.exceptions import EmbeddingError


class SparseEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    async def test_mixed_language_features_are_deterministic_and_normalized(self) -> None:
        embedding = HashedLexicalSparseEmbedding(hash_space=1_000_003)
        text = "数据库 Qdrant API_v2 /orders/123 数据库"

        first = embedding.embed_query(text)
        second = await embedding.aembed_query(text)

        self.assertEqual(first, second)
        self.assertEqual(tuple(sorted(set(first[0]))), tuple(first[0]))
        self.assertEqual(len(first[0]), len(first[1]))
        self.assertTrue(all(math.isfinite(value) for value in first[1]))
        self.assertAlmostEqual(sum(value * value for value in first[1]), 1.0)
        tokens = embedding.tokens(text)
        self.assertIn("数", tokens)
        self.assertIn("数据", tokens)
        self.assertIn("qdrant", tokens)
        self.assertIn("api_v2", tokens)

    async def test_empty_or_punctuation_only_input_is_rejected(self) -> None:
        embedding = HashedLexicalSparseEmbedding()
        with self.assertRaises(EmbeddingError):
            embedding.embed_query("   ")
        with self.assertRaises(EmbeddingError):
            embedding.embed_query("!!!")

    async def test_document_batch_matches_individual_encoding(self) -> None:
        embedding = HashedLexicalSparseEmbedding(include_cjk_bigrams=False)
        texts = ("退款 policy", "SKU-2048 库存")
        self.assertEqual(
            embedding.embed_documents(texts),
            tuple(embedding.embed_query(text) for text in texts),
        )


if __name__ == "__main__":
    unittest.main()
