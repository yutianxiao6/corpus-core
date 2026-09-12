from __future__ import annotations

import asyncio
import math
import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

from corpuscore.concurrency import AsyncMicroBatcher, QueuedReranker
from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.retrieval import RetrievalCandidate
from corpuscore.exceptions import LocalResourceMissingError, RerankerError
from corpuscore.rerankers import QwenCrossEncoderReranker


def candidate(chunk_id: str, score: float) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(
            chunk_id=chunk_id,
            document_id=f"doc-{chunk_id}",
            content=f"content {chunk_id}",
            embedding_text=f"content {chunk_id}",
            source_uri=f"file:///{chunk_id}.txt",
            chunk_index=0,
        ),
        final_score=score,
        rank=1,
        origins=["dense"],
    )


class ModelFixture:
    def __init__(self, scores: object) -> None:
        self.scores = scores
        self.inputs: Sequence[tuple[str, str]] = ()
        self.options: dict[str, object] = {}
        self.calls = 0

    def predict(self, inputs: Sequence[tuple[str, str]], **kwargs: object) -> object:
        self.calls += 1
        self.inputs = inputs
        self.options = kwargs
        return self.scores


class RerankerTests(unittest.IsolatedAsyncioTestCase):
    async def test_sigmoid_scores_sort_without_mutating_recall_candidates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ModelFixture([-2.0, 3.0, 0.0])
            reranker = QwenCrossEncoderReranker(
                directory,
                model=model,
                batch_size=2,
                maximum_candidates=3,
            )
            original = [candidate("a", 0.9), candidate("b", 0.8), candidate("c", 0.7)]

            result = reranker.rerank("query", original, top_n=3)
            async_result = await reranker.arerank("query", original, top_n=3)

        self.assertEqual([item.chunk.chunk_id for item in result], ["b", "c", "a"])
        self.assertEqual([item.rank for item in result], [1, 2, 3])
        self.assertAlmostEqual(result[0].rerank_score or 0, 1 / (1 + math.exp(-3)))
        self.assertEqual([item.chunk.chunk_id for item in async_result], ["b", "c", "a"])
        self.assertEqual([item.rerank_score for item in original], [None, None, None])
        self.assertEqual(model.inputs[0], ("query", "content a"))
        self.assertEqual(model.options["batch_size"], 2)

    async def test_candidate_limit_and_raw_score_mode(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reranker = QwenCrossEncoderReranker(
                directory,
                model=ModelFixture([5.0, 1.0]),
                maximum_candidates=2,
                score_mode="raw",
            )
            result = reranker.rerank(
                "query", [candidate("a", 1), candidate("b", 2), candidate("c", 3)]
            )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].rerank_score, 5.0)

    async def test_batch_flattens_pairs_into_one_model_prediction(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ModelFixture([0.1, 0.9, 0.8])
            reranker = QwenCrossEncoderReranker(
                directory, model=model, maximum_candidates=3, score_mode="raw"
            )

            results = reranker.rerank_batch(
                (
                    ("first", [candidate("a", 1), candidate("b", 2)], 2),
                    ("second", [candidate("c", 3)], 1),
                )
            )

        self.assertEqual(len(results), 2)
        self.assertEqual([item.chunk.chunk_id for item in results[0]], ["b", "a"])
        self.assertEqual([item.chunk.chunk_id for item in results[1]], ["c"])
        self.assertEqual(len(model.inputs), 3)
        self.assertEqual(model.inputs[2][0], "second")

    async def test_queued_reranking_microbatches_concurrent_requests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ModelFixture([0.1, 0.9, 0.8, 0.2])
            reranker = QwenCrossEncoderReranker(
                directory, model=model, maximum_candidates=2, score_mode="raw"
            )
            queued = QueuedReranker(
                reranker,
                AsyncMicroBatcher(
                    reranker.rerank_batch,
                    max_batch_size=4,
                    max_wait_ms=20,
                    workers=1,
                    queue_capacity=8,
                    enqueue_timeout_seconds=1,
                    execution_timeout_seconds=1,
                    name="reranker",
                ),
            )
            inputs = [candidate("a", 1), candidate("b", 2)]
            try:
                results = await asyncio.gather(
                    queued.arerank("first", inputs), queued.arerank("second", inputs)
                )
            finally:
                await queued.aclose()

        self.assertEqual(len(results), 2)
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(model.inputs), 4)

    async def test_invalid_model_output_is_wrapped(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            reranker = QwenCrossEncoderReranker(directory, model=ModelFixture([float("nan")]))
            with self.assertRaises(RerankerError):
                reranker.rerank("query", [candidate("a", 1)])

    async def test_missing_local_model_fails_without_loading(self) -> None:
        missing = Path(tempfile.gettempdir()) / "corpuscore-definitely-missing-reranker"
        with self.assertRaises(LocalResourceMissingError):
            QwenCrossEncoderReranker(missing)

    async def test_model_loader_is_forced_offline(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            model = ModelFixture([1.0])
            with patch("sentence_transformers.CrossEncoder", return_value=model) as loader:
                reranker = QwenCrossEncoderReranker(directory, revision="pinned", max_length=512)
                reranker.rerank("query", [candidate("a", 1)])

        options = loader.call_args.kwargs
        self.assertIs(options["local_files_only"], True)
        self.assertIs(options["trust_remote_code"], False)
        self.assertEqual(options["revision"], "pinned")
        self.assertEqual(options["max_length"], 512)


if __name__ == "__main__":
    unittest.main()
