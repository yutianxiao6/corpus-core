from __future__ import annotations

import unittest

from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.retrieval import RetrievalCandidate
from offline_rag.evaluation import RetrievalEvaluator, RetrievalExample


def result(identifier: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        Chunk(
            chunk_id=identifier,
            document_id="doc",
            content=identifier,
            embedding_text=identifier,
            source_uri="fixture.txt",
            chunk_index=0,
        )
    )


class EvaluationTests(unittest.TestCase):
    def test_recall_at_k_and_hit_rate(self) -> None:
        fixture = {
            "first": [result("a"), result("x")],
            "second": [result("missing")],
        }
        evaluator = RetrievalEvaluator(lambda query, k: fixture[query][:k])
        report = evaluator.evaluate(
            [
                RetrievalExample("first", ("a", "b")),
                RetrievalExample("second", ("c",)),
            ],
            k=2,
        )

        self.assertEqual(report.example_count, 2)
        self.assertEqual(report.recall_at_k, 0.25)
        self.assertEqual(report.hit_rate_at_k, 0.5)


if __name__ == "__main__":
    unittest.main()
