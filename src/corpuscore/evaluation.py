"""Small offline retrieval evaluation primitives for regression fixtures."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from corpuscore.contracts.retrieval import RetrievalCandidate
from corpuscore.exceptions import EvaluationRegressionError


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
    precision_at_k: float = 0.0
    mrr_at_k: float = 0.0
    ndcg_at_k: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        return {
            "example_count": self.example_count,
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "precision_at_k": self.precision_at_k,
            "mrr_at_k": self.mrr_at_k,
            "ndcg_at_k": self.ndcg_at_k,
        }


class RetrievalEvaluator:
    def __init__(self, retrieve: Callable[[str, int], Sequence[RetrievalCandidate]]) -> None:
        self._retrieve = retrieve

    def evaluate(self, examples: Sequence[RetrievalExample], *, k: int) -> EvaluationReport:
        if not examples:
            raise ValueError("examples must not be empty")
        if k <= 0:
            raise ValueError("k must be positive")
        recalls: list[float] = []
        precisions: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        hits = 0
        for example in examples:
            retrieved_ids = _unique_ids(self._retrieve(example.query, k), k)
            retrieved = set(retrieved_ids)
            relevant = set(example.relevant_chunk_ids)
            matched = len(retrieved.intersection(relevant))
            recalls.append(matched / len(relevant))
            precisions.append(matched / k)
            reciprocal_ranks.append(
                next(
                    (
                        1 / position
                        for position, item in enumerate(retrieved_ids, start=1)
                        if item in relevant
                    ),
                    0.0,
                )
            )
            dcg = sum(
                1 / math.log2(position + 1)
                for position, item in enumerate(retrieved_ids, start=1)
                if item in relevant
            )
            ideal_count = min(k, len(relevant))
            idcg = sum(1 / math.log2(position + 1) for position in range(1, ideal_count + 1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
            hits += matched > 0
        return EvaluationReport(
            example_count=len(examples),
            k=k,
            recall_at_k=sum(recalls) / len(recalls),
            hit_rate_at_k=hits / len(examples),
            precision_at_k=sum(precisions) / len(precisions),
            mrr_at_k=sum(reciprocal_ranks) / len(reciprocal_ranks),
            ndcg_at_k=sum(ndcgs) / len(ndcgs),
        )


def assert_thresholds(report: EvaluationReport, minimums: Mapping[str, float]) -> None:
    """Raise a machine-readable regression error when a metric misses its floor."""

    failures: dict[str, dict[str, float]] = {}
    for metric, minimum in minimums.items():
        if not hasattr(report, metric):
            raise ValueError(f"unknown evaluation metric: {metric}")
        actual = float(getattr(report, metric))
        if actual < minimum:
            failures[metric] = {"actual": actual, "minimum": minimum}
    if failures:
        raise EvaluationRegressionError(
            "evaluation metrics are below configured thresholds", details=failures
        )


def load_examples(path: str | Path) -> tuple[RetrievalExample, ...]:
    """Load a JSON array or JSONL list of ``{"query", "relevant_chunk_ids"}`` records."""

    source = Path(path)
    text = source.read_text(encoding="utf-8")
    if source.suffix.lower() == ".jsonl":
        raw: object = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        raw = json.loads(text)
    if not isinstance(raw, list):
        raise TypeError("evaluation fixture must contain a JSON array")
    examples: list[RetrievalExample] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise TypeError("each evaluation example must be an object")
        query = item.get("query")
        relevant = item.get("relevant_chunk_ids")
        if (
            not isinstance(query, str)
            or not isinstance(relevant, Sequence)
            or isinstance(relevant, (str, bytes))
        ):
            raise TypeError("evaluation example requires query and relevant_chunk_ids")
        if not all(isinstance(identifier, str) for identifier in relevant):
            raise ValueError("relevant_chunk_ids must contain strings")
        examples.append(RetrievalExample(query, tuple(relevant)))
    if not examples:
        raise ValueError("evaluation fixture must not be empty")
    return tuple(examples)


def _unique_ids(candidates: Sequence[RetrievalCandidate], k: int) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for candidate in candidates:
        chunk_id = candidate.chunk.chunk_id
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        result.append(chunk_id)
        if len(result) >= k:
            break
    return tuple(result)
