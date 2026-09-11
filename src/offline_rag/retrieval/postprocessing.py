"""Composable candidate filtering, MMR, diversity, and neighbor expansion."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import replace

from offline_rag.contracts.retrieval import RetrievalCandidate
from offline_rag.ports import EmbeddingProvider, Vector, VectorStorePort
from offline_rag.vectorstores.payloads import chunk_from_payload


def score_threshold_filter(
    candidates: Sequence[RetrievalCandidate], threshold: float | None
) -> tuple[RetrievalCandidate, ...]:
    if threshold is None:
        return tuple(candidates)
    filtered = [
        candidate
        for candidate in candidates
        if candidate.final_score is not None and candidate.final_score >= threshold
    ]
    return _ranked(filtered)


def limit_per_document(
    candidates: Sequence[RetrievalCandidate], maximum: int | None
) -> tuple[RetrievalCandidate, ...]:
    if maximum is None:
        return tuple(candidates)
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    counts: dict[str, int] = {}
    selected: list[RetrievalCandidate] = []
    for candidate in candidates:
        document_id = candidate.chunk.document_id
        if counts.get(document_id, 0) >= maximum:
            continue
        counts[document_id] = counts.get(document_id, 0) + 1
        selected.append(candidate)
    return _ranked(selected)


class MaximalMarginalRelevanceSelector:
    def __init__(self, embedding: EmbeddingProvider) -> None:
        self._embedding = embedding

    def select(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        *,
        limit: int,
        relevance_weight: float,
    ) -> tuple[RetrievalCandidate, ...]:
        if limit <= 0:
            raise ValueError("limit must be positive")
        if not 0 <= relevance_weight <= 1:
            raise ValueError("relevance_weight must be between 0 and 1")
        if not candidates:
            return ()
        query_vector = self._embedding.embed_query(query)
        vectors = tuple(
            self._embedding.embed_documents(
                [candidate.chunk.embedding_text for candidate in candidates]
            )
        )
        if len(vectors) != len(candidates):
            raise ValueError("candidate embedding count does not match candidate count")
        relevance = [_cosine(query_vector, vector) for vector in vectors]
        selected: list[int] = []
        remaining = set(range(len(candidates)))
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda index: (
                    relevance_weight * relevance[index]
                    - (1 - relevance_weight)
                    * max(
                        (_cosine(vectors[index], vectors[chosen]) for chosen in selected),
                        default=0.0,
                    ),
                    -index,
                ),
            )
            selected.append(best)
            remaining.remove(best)
        return _ranked([candidates[index] for index in selected])


class NeighborExpander:
    def __init__(self, vector_store: VectorStorePort) -> None:
        self._vector_store = vector_store

    def expand(
        self, candidates: Sequence[RetrievalCandidate], *, distance: int
    ) -> tuple[RetrievalCandidate, ...]:
        if distance < 0:
            raise ValueError("distance must be non-negative")
        if distance == 0 or not candidates:
            return tuple(candidates)
        cache = {candidate.chunk.chunk_id: candidate.chunk for candidate in candidates}
        previous_paths: list[list[str]] = [[] for _ in candidates]
        next_paths: list[list[str]] = [[] for _ in candidates]
        previous_frontier = [candidate.chunk.previous_id for candidate in candidates]
        next_frontier = [candidate.chunk.next_id for candidate in candidates]
        for _depth in range(distance):
            needed = tuple(
                dict.fromkeys(
                    chunk_id
                    for chunk_id in (*previous_frontier, *next_frontier)
                    if chunk_id is not None and chunk_id not in cache
                )
            )
            for hit in self._vector_store.fetch(needed):
                cache[hit.chunk_id] = chunk_from_payload(hit.payload)
            for position, chunk_id in enumerate(previous_frontier):
                chunk = cache.get(chunk_id) if chunk_id else None
                if chunk is None:
                    previous_frontier[position] = None
                    continue
                previous_paths[position].append(chunk.chunk_id)
                previous_frontier[position] = chunk.previous_id
            for position, chunk_id in enumerate(next_frontier):
                chunk = cache.get(chunk_id) if chunk_id else None
                if chunk is None:
                    next_frontier[position] = None
                    continue
                next_paths[position].append(chunk.chunk_id)
                next_frontier[position] = chunk.next_id
        primary_ids = {candidate.chunk.chunk_id for candidate in candidates}
        seen: set[str] = set()
        expanded: list[RetrievalCandidate] = []
        for position, candidate in enumerate(candidates):
            ordered_ids = (
                *reversed(previous_paths[position]),
                candidate.chunk.chunk_id,
                *next_paths[position],
            )
            for chunk_id in ordered_ids:
                if chunk_id in seen or (
                    chunk_id in primary_ids and chunk_id != candidate.chunk.chunk_id
                ):
                    continue
                seen.add(chunk_id)
                if chunk_id == candidate.chunk.chunk_id:
                    expanded.append(candidate)
                else:
                    expanded.append(
                        RetrievalCandidate(
                            chunk=cache[chunk_id],
                            final_score=None,
                            rank=None,
                            origins=["neighbor"],
                        )
                    )
        return tuple(expanded)


def _ranked(candidates: Sequence[RetrievalCandidate]) -> tuple[RetrievalCandidate, ...]:
    return tuple(
        replace(candidate, rank=rank, origins=list(candidate.origins))
        for rank, candidate in enumerate(candidates, start=1)
    )


def _cosine(left: Vector, right: Vector) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine vectors must have the same non-zero dimension")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)
