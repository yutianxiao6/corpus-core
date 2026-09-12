from __future__ import annotations

import unittest
from collections.abc import Sequence
from typing import ClassVar

from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.indexing import EmbeddingSpecification
from corpuscore.contracts.retrieval import RetrievalCandidate, SearchHit
from corpuscore.ports import Vector
from corpuscore.retrieval import (
    MaximalMarginalRelevanceSelector,
    NeighborExpander,
    limit_per_document,
    score_threshold_filter,
)
from corpuscore.vectorstores import chunk_to_payload


def make_candidate(
    chunk_id: str,
    score: float,
    *,
    document_id: str | None = None,
    previous_id: str | None = None,
    next_id: str | None = None,
) -> RetrievalCandidate:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id=document_id or f"doc-{chunk_id}",
        content=chunk_id,
        embedding_text=chunk_id,
        source_uri=f"file:///{chunk_id}.txt",
        chunk_index=0,
        previous_id=previous_id,
        next_id=next_id,
    )
    return RetrievalCandidate(chunk=chunk, final_score=score, rank=1, origins=["dense"])


class EmbeddingFixture:
    specification: EmbeddingSpecification

    vectors: ClassVar[dict[str, Vector]] = {
        "query": (1.0, 0.0),
        "similar-a": (1.0, 0.0),
        "similar-b": (0.99, 0.01),
        "diverse": (0.6, 0.8),
    }

    def embed_query(self, text: str) -> Vector:
        return self.vectors[text]

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple(self.vectors[text] for text in texts)


class StoreFixture:
    def __init__(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.requests: list[tuple[str, ...]] = []

    def fetch(self, chunk_ids: Sequence[str]) -> Sequence[SearchHit]:
        self.requests.append(tuple(chunk_ids))
        return tuple(
            SearchHit(chunk_id, 0.0, chunk_to_payload(self.chunks[chunk_id]))
            for chunk_id in chunk_ids
            if chunk_id in self.chunks
        )

    async def afetch(self, chunk_ids: Sequence[str]) -> Sequence[SearchHit]:
        return self.fetch(chunk_ids)


class RetrievalPostprocessingTests(unittest.TestCase):
    def test_threshold_and_per_document_limit_reset_ranks(self) -> None:
        candidates = [
            make_candidate("a", 0.9, document_id="same"),
            make_candidate("b", 0.8, document_id="same"),
            make_candidate("c", 0.7, document_id="other"),
            make_candidate("low", 0.2, document_id="third"),
        ]

        filtered = score_threshold_filter(candidates, 0.5)
        diverse = limit_per_document(filtered, 1)

        self.assertEqual([item.chunk.chunk_id for item in diverse], ["a", "c"])
        self.assertEqual([item.rank for item in diverse], [1, 2])

    def test_mmr_trades_small_relevance_loss_for_non_redundancy(self) -> None:
        candidates = [
            make_candidate("similar-a", 0.9),
            make_candidate("similar-b", 0.89),
            make_candidate("diverse", 0.8),
        ]

        result = MaximalMarginalRelevanceSelector(EmbeddingFixture()).select(
            "query", candidates, limit=2, relevance_weight=0.4
        )

        self.assertEqual([item.chunk.chunk_id for item in result], ["similar-a", "diverse"])

    def test_neighbor_expansion_preserves_document_order_without_fake_scores(self) -> None:
        chunks = [
            make_candidate("one", 0, next_id="two").chunk,
            make_candidate("two", 0, previous_id="one", next_id="three").chunk,
            make_candidate("three", 0, previous_id="two", next_id="four").chunk,
            make_candidate("four", 0, previous_id="three").chunk,
        ]
        store = StoreFixture(chunks)
        seed = make_candidate("three", 0.95, previous_id="two", next_id="four")

        result = NeighborExpander(store).expand([seed], distance=2)

        self.assertEqual([item.chunk.chunk_id for item in result], ["one", "two", "three", "four"])
        neighbors = [item for item in result if item.chunk.chunk_id != "three"]
        self.assertTrue(all(item.final_score is None for item in neighbors))
        self.assertTrue(all(item.rank is None for item in neighbors))
        self.assertTrue(all(item.origins == ["neighbor"] for item in neighbors))
        self.assertLessEqual(len(store.requests), 2)


class AsyncRetrievalPostprocessingTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_neighbor_expansion_matches_sync_order(self) -> None:
        chunks = [
            make_candidate("one", 0, next_id="two").chunk,
            make_candidate("two", 0, previous_id="one", next_id="three").chunk,
            make_candidate("three", 0, previous_id="two").chunk,
        ]
        store = StoreFixture(chunks)
        seed = make_candidate("two", 0.95, previous_id="one", next_id="three")

        result = await NeighborExpander(store).aexpand([seed], distance=1)

        self.assertEqual([item.chunk.chunk_id for item in result], ["one", "two", "three"])


if __name__ == "__main__":
    unittest.main()
