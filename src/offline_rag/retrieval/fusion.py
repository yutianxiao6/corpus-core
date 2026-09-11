"""Deterministic rank fusion for dense and sparse result lists."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from offline_rag.contracts.common import JSONValue
from offline_rag.contracts.retrieval import SearchHit


@dataclass(frozen=True, slots=True)
class FusedSearchHit:
    chunk_id: str
    payload: Mapping[str, JSONValue]
    score: float
    dense_score: float | None
    sparse_score: float | None
    origins: tuple[str, ...]


def reciprocal_rank_fusion(
    dense_hits: Sequence[SearchHit],
    sparse_hits: Sequence[SearchHit],
    *,
    constant: int = 60,
    dense_weight: float = 1.0,
    sparse_weight: float = 1.0,
) -> tuple[FusedSearchHit, ...]:
    """Fuse rankings by weighted reciprocal rank, deduplicated by chunk id."""

    if constant <= 0:
        raise ValueError("constant must be positive")
    _validate_weights(dense_weight, sparse_weight)
    scores: dict[str, float] = {}
    for hits, weight in ((dense_hits, dense_weight), (sparse_hits, sparse_weight)):
        seen: set[str] = set()
        for rank, hit in enumerate(hits, start=1):
            if hit.chunk_id in seen:
                continue
            seen.add(hit.chunk_id)
            scores[hit.chunk_id] = scores.get(hit.chunk_id, 0.0) + weight / (constant + rank)
    return _materialize(scores, dense_hits, sparse_hits)


def normalized_weighted_fusion(
    dense_hits: Sequence[SearchHit],
    sparse_hits: Sequence[SearchHit],
    *,
    dense_weight: float = 0.5,
    sparse_weight: float = 0.5,
) -> tuple[FusedSearchHit, ...]:
    """Min-max normalize each result list before applying configured weights."""

    _validate_weights(dense_weight, sparse_weight)
    dense_scores = _normalized_scores(dense_hits)
    sparse_scores = _normalized_scores(sparse_hits)
    chunk_ids = set(dense_scores) | set(sparse_scores)
    scores = {
        chunk_id: dense_weight * dense_scores.get(chunk_id, 0.0)
        + sparse_weight * sparse_scores.get(chunk_id, 0.0)
        for chunk_id in chunk_ids
    }
    return _materialize(scores, dense_hits, sparse_hits)


def _validate_weights(dense_weight: float, sparse_weight: float) -> None:
    if dense_weight < 0 or sparse_weight < 0:
        raise ValueError("fusion weights must be non-negative")
    if dense_weight + sparse_weight <= 0:
        raise ValueError("at least one fusion weight must be positive")


def _normalized_scores(hits: Sequence[SearchHit]) -> dict[str, float]:
    best = _best_hits(hits)
    if not best:
        return {}
    values = [hit.score for hit in best.values()]
    minimum, maximum = min(values), max(values)
    if minimum == maximum:
        return {chunk_id: 1.0 for chunk_id in best}
    scale = maximum - minimum
    return {chunk_id: (hit.score - minimum) / scale for chunk_id, hit in best.items()}


def _best_hits(hits: Sequence[SearchHit]) -> dict[str, SearchHit]:
    result: dict[str, SearchHit] = {}
    for hit in hits:
        previous = result.get(hit.chunk_id)
        if previous is None or hit.score > previous.score:
            result[hit.chunk_id] = hit
    return result


def _materialize(
    scores: Mapping[str, float],
    dense_hits: Sequence[SearchHit],
    sparse_hits: Sequence[SearchHit],
) -> tuple[FusedSearchHit, ...]:
    dense = _best_hits(dense_hits)
    sparse = _best_hits(sparse_hits)
    result = [
        FusedSearchHit(
            chunk_id=chunk_id,
            payload=(dense.get(chunk_id) or sparse[chunk_id]).payload,
            score=score,
            dense_score=dense[chunk_id].score if chunk_id in dense else None,
            sparse_score=sparse[chunk_id].score if chunk_id in sparse else None,
            origins=tuple(
                origin
                for origin, values in (("dense", dense), ("sparse", sparse))
                if chunk_id in values
            ),
        )
        for chunk_id, score in scores.items()
    ]
    return tuple(sorted(result, key=lambda item: (-item.score, item.chunk_id)))
