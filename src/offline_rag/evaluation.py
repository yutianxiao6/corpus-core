"""Small offline retrieval evaluation primitives for regression fixtures."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from offline_rag.contracts.retrieval import RetrievalCandidate


@dataclass(frozen=True, slots=True)
class RetrievalExample:
    query: str
    relevant_chunk_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.query.strip():
            raise ValueError("evaluation query must not be empty")
        if not self.relevant_chunk_ids:
            raise ValueError("relevant_chunk_ids must not be empty")


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    example_count: int
    k: int
    recall_at_k: float
    hit_rate_at_k: float


class RetrievalEvaluator:
    def __init__(self, retrieve: Callable[[str, int], Sequence[RetrievalCandidate]]) -> None:
        self._retrieve = retrieve

    def evaluate(self, examples: Sequence[RetrievalExample], *, k: int) -> EvaluationReport:
        if not examples:
            raise ValueError("examples must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        recalls: list[float] = []
        hits = 0
        for example in examples:
            retrieved = {
                candidate.chunk.chunk_id for candidate in self._retrieve(example.query, k)[:k]
            }
            relevant = set(example.relevant_chunk_ids)
            matched = len(retrieved.intersection(relevant))
            recalls.append(matched / len(relevant))
            hits += matched > 0
        return EvaluationReport(
            example_count=len(examples),
            k=k,
            recall_at_k=sum(recalls) / len(recalls),
            hit_rate_at_k=hits / len(examples),
        )
