"""Sparse lexical retrieval."""

from __future__ import annotations

from collections.abc import Sequence

from corpuscore.contracts.retrieval import (
    RetrievalCandidate,
    RetrievalRequest,
    SearchHit,
    SearchRequest,
    SearchVector,
)
from corpuscore.ports import SparseEmbeddingProvider, SparseVector, VectorStorePort
from corpuscore.vectorstores.payloads import chunk_from_payload


class SparseSimilarityStrategy:
    def __init__(
        self,
        embedding: SparseEmbeddingProvider,
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
        sparse = self._embedding.embed_query(request.query)
        return self._candidates(request, self._vector_store.search(self._request(request, sparse)))

    async def aretrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]:
        sparse = await self._embedding.aembed_query(request.query)
        hits = await self._vector_store.asearch(self._request(request, sparse))
        return self._candidates(request, hits)

    def _request(self, request: RetrievalRequest, sparse: SparseVector) -> SearchRequest:
        return SearchRequest(
            (),
            limit=max(self._fetch_k, request.options.final_k),
            filters=request.options.filters,
            sparse_indices=sparse[0],
            sparse_values=sparse[1],
            vector=SearchVector.SPARSE,
        )

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
                    sparse_score=hit.score,
                    final_score=hit.score,
                    rank=len(candidates) + 1,
                    origins=["sparse"],
                )
            )
            if len(candidates) >= request.options.final_k:
                break
        return tuple(candidates)
