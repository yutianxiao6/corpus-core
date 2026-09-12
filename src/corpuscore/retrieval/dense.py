"""Dense-vector similarity retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from corpuscore.contracts.retrieval import (
    RetrievalCandidate,
    RetrievalRequest,
    SearchHit,
    SearchRequest,
)
from corpuscore.ports import EmbeddingProvider, VectorStorePort
from corpuscore.vectorstores.payloads import chunk_from_payload


class DenseSimilarityStrategy:
    def __init__(
        self,
        embedding: EmbeddingProvider,
        vector_store: VectorStorePort,
        *,
        fetch_k: int = 20,
    ) -> None:
        if fetch_k <= 0:
            raise ValueError("fetch_k must be positive")
        self._embedding = embedding
        self._vector_store = vector_store
        self._fetch_k = fetch_k

    def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]:
        vector = self._embedding.embed_query(request.query)
        hits = self._vector_store.search(self._search_request(request, vector))
        return self._candidates(request, hits)

    async def aretrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]:
        vector = await self._embedding.aembed_query(request.query)
        hits = await self._vector_store.asearch(self._search_request(request, vector))
        return self._candidates(request, hits)

    def _search_request(self, request: RetrievalRequest, vector: Sequence[float]) -> SearchRequest:
        limit = max(self._fetch_k, request.options.final_k)
        return SearchRequest(query_vector=vector, limit=limit, filters=request.options.filters)

    @staticmethod
    def _candidates(
        request: RetrievalRequest, hits: Sequence[SearchHit]
    ) -> tuple[RetrievalCandidate, ...]:
        candidates: list[RetrievalCandidate] = []
        seen: set[str] = set()
        for hit in sorted(hits, key=lambda item: item.score, reverse=True):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            candidates.append(
                RetrievalCandidate(
                    chunk=chunk_from_payload(hit.payload),
                    dense_score=hit.score,
                    final_score=hit.score,
                    rank=len(candidates) + 1,
                    origins=["dense"],
                )
            )
            if len(candidates) >= request.options.final_k:
                break
        return tuple(candidates)
