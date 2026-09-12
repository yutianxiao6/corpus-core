"""Dense plus sparse retrieval with configurable local fusion."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Literal

from offline_rag.contracts.retrieval import (
    RetrievalCandidate,
    RetrievalRequest,
    SearchHit,
    SearchRequest,
    SearchVector,
)
from offline_rag.ports import EmbeddingProvider, SparseEmbeddingProvider, VectorStorePort
from offline_rag.retrieval.fusion import (
    FusedSearchHit,
    normalized_weighted_fusion,
    reciprocal_rank_fusion,
)
from offline_rag.vectorstores.payloads import chunk_from_payload


class HybridRetrievalStrategy:
    def __init__(
        self,
        dense_embedding: EmbeddingProvider,
        sparse_embedding: SparseEmbeddingProvider,
        vector_store: VectorStorePort,
        *,
        dense_fetch_k: int = 20,
        sparse_fetch_k: int = 20,
        fusion: Literal["rrf", "weighted"] = "rrf",
        rrf_constant: int = 60,
        dense_weight: float = 0.5,
        sparse_weight: float = 0.5,
    ) -> None:
        if dense_fetch_k <= 0 or sparse_fetch_k <= 0:
            raise ValueError("fetch limits must be positive")
        if fusion not in ("rrf", "weighted"):
            raise ValueError(f"unsupported fusion type: {fusion}")
        self._dense_embedding = dense_embedding
        self._sparse_embedding = sparse_embedding
        self._vector_store = vector_store
        self._dense_fetch_k = dense_fetch_k
        self._sparse_fetch_k = sparse_fetch_k
        self._fusion = fusion
        self._rrf_constant = rrf_constant
        self._dense_weight = dense_weight
        self._sparse_weight = sparse_weight

    def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]:
        dense = self._dense_embedding.embed_query(request.query)
        sparse = self._sparse_embedding.embed_query(request.query)
        dense_hits = self._vector_store.search(
            SearchRequest(dense, self._dense_limit(request), request.options.filters)
        )
        sparse_hits = self._vector_store.search(
            SearchRequest(
                (),
                self._sparse_limit(request),
                request.options.filters,
                sparse[0],
                sparse[1],
                SearchVector.SPARSE,
            )
        )
        return self._candidates(request, self._fuse(dense_hits, sparse_hits))

    async def aretrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]:
        dense, sparse = await asyncio.gather(
            self._dense_embedding.aembed_query(request.query),
            self._sparse_embedding.aembed_query(request.query),
        )
        dense_hits, sparse_hits = await asyncio.gather(
            self._vector_store.asearch(
                SearchRequest(dense, self._dense_limit(request), request.options.filters)
            ),
            self._vector_store.asearch(
                SearchRequest(
                    (),
                    self._sparse_limit(request),
                    request.options.filters,
                    sparse[0],
                    sparse[1],
                    SearchVector.SPARSE,
                )
            ),
        )
        return self._candidates(request, self._fuse(dense_hits, sparse_hits))

    def _dense_limit(self, request: RetrievalRequest) -> int:
        return max(self._dense_fetch_k, request.options.final_k)

    def _sparse_limit(self, request: RetrievalRequest) -> int:
        return max(self._sparse_fetch_k, request.options.final_k)

    def _fuse(
        self, dense_hits: Sequence[SearchHit], sparse_hits: Sequence[SearchHit]
    ) -> tuple[FusedSearchHit, ...]:
        if self._fusion == "rrf":
            return reciprocal_rank_fusion(
                dense_hits,
                sparse_hits,
                constant=self._rrf_constant,
                dense_weight=self._dense_weight,
                sparse_weight=self._sparse_weight,
            )
        return normalized_weighted_fusion(
            dense_hits,
            sparse_hits,
            dense_weight=self._dense_weight,
            sparse_weight=self._sparse_weight,
        )

    @staticmethod
    def _candidates(
        request: RetrievalRequest, hits: Sequence[FusedSearchHit]
    ) -> tuple[RetrievalCandidate, ...]:
        result: list[RetrievalCandidate] = []
        for hit in hits:
            result.append(
                RetrievalCandidate(
                    chunk=chunk_from_payload(hit.payload),
                    dense_score=hit.dense_score,
                    sparse_score=hit.sparse_score,
                    fusion_score=hit.score,
                    final_score=hit.score,
                    rank=len(result) + 1,
                    origins=list(hit.origins),
                )
            )
            if len(result) >= request.options.final_k:
                break
        return tuple(result)
