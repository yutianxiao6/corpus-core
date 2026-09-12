from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.retrieval import RetrievalCandidate
from corpuscore.evaluation import (
    RetrievalEvaluator,
    RetrievalExample,
    assert_thresholds,
    load_examples,
)
from corpuscore.exceptions import EvaluationRegressionError


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
        self.assertEqual(report.precision_at_k, 0.25)
        self.assertEqual(report.mrr_at_k, 0.5)
        self.assertAlmostEqual(report.ndcg_at_k, 1 / (1 + 1 / math.log2(3)) / 2)

    def test_metrics_can_be_gated_and_fail_as_machine_readable_error(self) -> None:
        report = RetrievalEvaluator(lambda query, k: [result("wrong")]).evaluate(
            [RetrievalExample("query", ("expected",))], k=1
        )
        assert_thresholds(report, {"recall_at_k": 0.0, "mrr_at_k": 0.0})
        with self.assertRaises(EvaluationRegressionError) as raised:
            assert_thresholds(report, {"recall_at_k": 0.1})
        self.assertEqual(raised.exception.code, "evaluation_regression")

    def test_load_json_and_jsonl_fixtures(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "eval.json"
            json_path.write_text(
                '[{"query": "退款", "relevant_chunk_ids": ["chunk-1"]}]',
                encoding="utf-8",
            )
            jsonl_path = root / "eval.jsonl"
            jsonl_path.write_text(
                '{"query": "安装", "relevant_chunk_ids": ["chunk-2"]}\n',
                encoding="utf-8",
            )
            self.assertEqual(load_examples(json_path)[0].query, "退款")
            self.assertEqual(load_examples(jsonl_path)[0].relevant_chunk_ids, ("chunk-2",))


if __name__ == "__main__":
    unittest.main()
