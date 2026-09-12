from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path

from offline_rag.concurrency import AsyncMicroBatcher, QueuedEmbeddingProvider
from offline_rag.embeddings import QwenSentenceTransformerEmbedding
from offline_rag.exceptions import EmbeddingError, OfflineResourceMissingError


class FakeSentenceModel:
    def __init__(self, dimension: int = 3) -> None:
        self.dimension = dimension
        self.max_seq_length = 0
        self.calls: list[tuple[object, dict[str, object]]] = []

    def encode(self, inputs: object, **kwargs: object) -> object:
        self.calls.append((inputs, kwargs))
        count = len(inputs) if isinstance(inputs, (list, tuple)) else 1
        return [[float(index) for index in range(self.dimension)] for _ in range(count)]

    def preprocess(self, inputs: list[str], **kwargs: object) -> dict[str, object]:
        return {"input_ids": [[index for index, _ in enumerate(inputs[0].split(), start=1)]]}


class EmbeddingAdapterTests(unittest.TestCase):
    def _model_dir(self, directory: str) -> Path:
        path = Path(directory) / "Qwen3-Embedding-0.6B"
        path.mkdir()
        (path / "config.json").write_text('{"model_type":"qwen3"}', encoding="utf-8")
        return path

    def test_documents_have_no_prompt_and_queries_have_instruction_prompt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fake = FakeSentenceModel()
            adapter = QwenSentenceTransformerEmbedding(
                self._model_dir(directory),
                dimension=3,
                max_length=128,
                query_instruction="检索回答问题的段落。",
                model=fake,
            )
            documents = adapter.embed_documents(["文档一", "document two"])
            queries = adapter.embed_queries(["如何安装？", "How to install?"])
            query = queries[0]

        self.assertEqual(len(documents), 2)
        self.assertEqual(len(query), 3)
        self.assertEqual(len(queries), 2)
        self.assertNotIn("prompt", fake.calls[0][1])
        self.assertEqual(fake.calls[1][1]["prompt"], "Instruct: 检索回答问题的段落。\nQuery: ")
        self.assertEqual(fake.max_seq_length, 128)
        self.assertEqual(adapter.count_tokens("one two three"), 3)

    def test_specification_contains_deterministic_local_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = self._model_dir(directory)
            first = QwenSentenceTransformerEmbedding(path, dimension=3, model=FakeSentenceModel())
            second = QwenSentenceTransformerEmbedding(path, dimension=3, model=FakeSentenceModel())

        self.assertEqual(first.specification.model_checksum, second.specification.model_checksum)
        self.assertEqual(first.specification.fingerprint(), second.specification.fingerprint())
        self.assertEqual(len(first.specification.model_checksum), 64)

    def test_dimension_mismatch_and_empty_input_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            adapter = QwenSentenceTransformerEmbedding(
                self._model_dir(directory), dimension=4, model=FakeSentenceModel(dimension=3)
            )
            with self.assertRaisesRegex(EmbeddingError, "dimension"):
                adapter.embed_documents(["content"])
            with self.assertRaises(EmbeddingError):
                adapter.embed_query(" ")

    def test_missing_or_empty_local_model_is_rejected_without_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "missing"
            with self.assertRaises(OfflineResourceMissingError):
                QwenSentenceTransformerEmbedding(missing)
            empty = Path(directory) / "empty"
            empty.mkdir()
            with self.assertRaisesRegex(OfflineResourceMissingError, "empty"):
                QwenSentenceTransformerEmbedding(empty)

    def test_non_finite_vectors_are_rejected(self) -> None:
        class NonFiniteModel(FakeSentenceModel):
            def encode(self, inputs: object, **kwargs: object) -> object:
                return [[0.0, float("nan"), 1.0]]

        with tempfile.TemporaryDirectory() as directory:
            adapter = QwenSentenceTransformerEmbedding(
                self._model_dir(directory), dimension=3, model=NonFiniteModel()
            )
            with self.assertRaisesRegex(EmbeddingError, "non-finite"):
                adapter.embed_query("query")


class AsyncEmbeddingAdapterTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_query_matches_sync_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            path.mkdir()
            (path / "config.json").write_text("{}", encoding="utf-8")
            adapter = QwenSentenceTransformerEmbedding(path, dimension=3, model=FakeSentenceModel())
            vector = await adapter.aembed_query("query")
        self.assertEqual(tuple(vector), (0.0, 1.0, 2.0))

    async def test_queued_queries_share_one_model_encode_call(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model"
            path.mkdir()
            (path / "config.json").write_text("{}", encoding="utf-8")
            model = FakeSentenceModel()
            adapter = QwenSentenceTransformerEmbedding(path, dimension=3, model=model)
            queued = QueuedEmbeddingProvider(
                adapter,
                AsyncMicroBatcher(
                    adapter.embed_queries,
                    max_batch_size=4,
                    max_wait_ms=20,
                    workers=1,
                    queue_capacity=8,
                    enqueue_timeout_seconds=1,
                    execution_timeout_seconds=1,
                    name="embedding",
                ),
            )
            try:
                vectors = await asyncio.gather(
                    queued.aembed_query("first"), queued.aembed_query("second")
                )
            finally:
                await queued.aclose()

        self.assertEqual(len(vectors), 2)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(model.calls[0][0], ("first", "second"))


if __name__ == "__main__":
    unittest.main()
