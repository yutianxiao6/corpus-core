from __future__ import annotations

import unittest
from collections.abc import Sequence

from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.indexing import EmbeddingSpecification
from offline_rag.contracts.retrieval import (
    RetrievalOptions,
    RetrievalRequest,
    SearchHit,
    SearchRequest,
)
from offline_rag.ports import Vector
from offline_rag.retrieval import DenseSimilarityStrategy
from offline_rag.vectorstores import chunk_to_payload


class FakeEmbedding:
    specification: EmbeddingSpecification

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple((1.0, 0.0) for _ in texts)

    def embed_query(self, text: str) -> Vector:
        return (1.0, 0.0)

    async def aembed_query(self, text: str) -> Vector:
        return self.embed_query(text)


class FakeStore:
    def __init__(self, hits: Sequence[SearchHit]) -> None:
        self.hits = hits
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> Sequence[SearchHit]:
        self.requests.append(request)
        return self.hits

    async def asearch(self, request: SearchRequest) -> Sequence[SearchHit]:
        return self.search(request)


def make_hit(chunk_id: str, score: float) -> SearchHit:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=f"content {chunk_id}",
        embedding_text=f"content {chunk_id}",
        source_uri=f"file:///{chunk_id}.txt",
        chunk_index=0,
    )
    return SearchHit(chunk_id, score, chunk_to_payload(chunk))


class DenseRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_strategy_defers_threshold_dedupes_and_preserves_async_parity(self) -> None:
        hits = [make_hit("low", 0.2), make_hit("best", 0.9), make_hit("best", 0.8)]
        store = FakeStore(hits)
        strategy = DenseSimilarityStrategy(FakeEmbedding(), store, fetch_k=10)
        request = RetrievalRequest(
            "query",
            RetrievalOptions(
                profile="fast",
                organizer="flat",
                final_k=2,
                score_threshold=0.5,
                filters={"department": "engineering"},
            ),
        )

        sync_result = strategy.retrieve(request)
        async_result = await strategy.aretrieve(request)

        # Thresholding belongs to the post-rerank pipeline. Retrieval strategies
        # must preserve low-scoring recall candidates for a reranker to promote.
        self.assertEqual([item.chunk.chunk_id for item in sync_result], ["best", "low"])
        self.assertEqual(sync_result[0].rank, 1)
        self.assertEqual(sync_result[0].dense_score, 0.9)
        self.assertEqual(sync_result[0].origins, ["dense"])
        self.assertEqual(
            [item.chunk.chunk_id for item in async_result],
            [item.chunk.chunk_id for item in sync_result],
        )
        self.assertEqual(store.requests[0].limit, 10)
        self.assertEqual(store.requests[0].filters["department"], "engineering")


if __name__ == "__main__":
    unittest.main()
