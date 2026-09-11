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
    SearchVector,
)
from offline_rag.embeddings import HashedLexicalSparseEmbedding
from offline_rag.ports import Vector
from offline_rag.retrieval import HybridRetrievalStrategy
from offline_rag.vectorstores import chunk_to_payload


class DenseFixture:
    specification: EmbeddingSpecification

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple((1.0, 0.0) for _ in texts)

    def embed_query(self, text: str) -> Vector:
        return (1.0, 0.0)

    async def aembed_query(self, text: str) -> Vector:
        return self.embed_query(text)


def make_hit(chunk_id: str, score: float) -> SearchHit:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=f"doc-{chunk_id}",
        content=chunk_id,
        embedding_text=chunk_id,
        source_uri=f"file:///{chunk_id}.txt",
        chunk_index=0,
    )
    return SearchHit(chunk_id, score, chunk_to_payload(chunk))


class StoreFixture:
    def __init__(self) -> None:
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> Sequence[SearchHit]:
        self.requests.append(request)
        if request.vector is SearchVector.SPARSE:
            return (make_hit("shared", 20.0), make_hit("lexical", 18.0))
        return (make_hit("semantic", 0.95), make_hit("shared", 0.8))

    async def asearch(self, request: SearchRequest) -> Sequence[SearchHit]:
        return self.search(request)


class HybridRetrievalTests(unittest.IsolatedAsyncioTestCase):
    async def test_sync_and_async_hybrid_results_match(self) -> None:
        store = StoreFixture()
        strategy = HybridRetrievalStrategy(
            DenseFixture(),
            HashedLexicalSparseEmbedding(),
            store,
            dense_fetch_k=8,
            sparse_fetch_k=12,
            fusion="rrf",
        )
        request = RetrievalRequest(
            "退款 policy",
            RetrievalOptions(final_k=2, filters={"department": "support"}),
        )

        sync_result = strategy.retrieve(request)
        async_result = await strategy.aretrieve(request)

        self.assertEqual([item.chunk.chunk_id for item in sync_result], ["shared", "semantic"])
        self.assertEqual(
            [item.chunk.chunk_id for item in async_result],
            [item.chunk.chunk_id for item in sync_result],
        )
        self.assertEqual(sync_result[0].origins, ["dense", "sparse"])
        self.assertIsNotNone(sync_result[0].dense_score)
        self.assertIsNotNone(sync_result[0].sparse_score)
        self.assertEqual(store.requests[0].limit, 8)
        self.assertEqual(store.requests[1].limit, 12)
        self.assertEqual(store.requests[1].vector, SearchVector.SPARSE)
        self.assertEqual(store.requests[1].filters["department"], "support")


if __name__ == "__main__":
    unittest.main()
