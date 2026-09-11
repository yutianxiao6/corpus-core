"""Query, candidate, citation, and retrieval result contracts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.common import (
    JSONValue,
    freeze_metadata,
    require_non_empty,
    require_positive,
)


@dataclass(frozen=True, slots=True)
class ContextBudget:
    max_tokens: int | None = None
    max_characters: int | None = None
    maximum_chunks_per_document: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens is None and self.max_characters is None:
            raise ValueError("at least one context budget limit must be configured")
        for name in ("max_tokens", "max_characters", "maximum_chunks_per_document"):
            value = getattr(self, name)
            if value is not None:
                require_positive(value, name)


@dataclass(frozen=True, slots=True)
class RetrievalOptions:
    profile: str = "balanced"
    final_k: int = 6
    score_threshold: float | None = None
    organizer: str = "context"
    debug: bool = False
    filters: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.profile, "profile")
        require_non_empty(self.organizer, "organizer")
        require_positive(self.final_k, "final_k")
        object.__setattr__(self, "filters", freeze_metadata(self.filters))


@dataclass(frozen=True, slots=True)
class QueryOverrides:
    filters: Mapping[str, JSONValue] = field(default_factory=dict)
    final_k: int | None = None
    score_threshold: float | None = None
    organizer: str | None = None
    rerank_top_n: int | None = None
    mmr_lambda: float | None = None
    mmr_fetch_k: int | None = None
    neighbor_expansion: int | None = None
    maximum_chunks_per_document: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "final_k",
            "rerank_top_n",
            "mmr_fetch_k",
            "maximum_chunks_per_document",
        ):
            value = getattr(self, name)
            if value is not None:
                require_positive(value, name)
        if self.neighbor_expansion is not None and self.neighbor_expansion < 0:
            raise ValueError("neighbor_expansion must be non-negative")
        if self.mmr_lambda is not None and not 0 <= self.mmr_lambda <= 1:
            raise ValueError("mmr_lambda must be between 0 and 1")
        if self.organizer is not None:
            require_non_empty(self.organizer, "organizer")
        object.__setattr__(self, "filters", freeze_metadata(self.filters))


@dataclass(frozen=True, slots=True)
class RetrievalRequest:
    query: str
    options: RetrievalOptions = field(default_factory=RetrievalOptions)

    def __post_init__(self) -> None:
        require_non_empty(self.query, "query")


class SearchVector(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass(frozen=True, slots=True)
class SearchRequest:
    query_vector: Sequence[float]
    limit: int
    filters: Mapping[str, JSONValue] = field(default_factory=dict)
    sparse_indices: Sequence[int] = ()
    sparse_values: Sequence[float] = ()
    vector: SearchVector = SearchVector.DENSE

    def __post_init__(self) -> None:
        require_positive(self.limit, "limit")
        if len(self.sparse_indices) != len(self.sparse_values):
            raise ValueError("sparse vector indices and values must have equal length")
        if tuple(self.sparse_indices) != tuple(sorted(set(self.sparse_indices))):
            raise ValueError("sparse vector indices must be sorted and unique")
        if any(index < 0 for index in self.sparse_indices):
            raise ValueError("sparse vector indices must be non-negative")
        if self.vector is SearchVector.DENSE and not self.query_vector:
            raise ValueError("dense search requires query_vector")
        if self.vector is SearchVector.SPARSE and not self.sparse_indices:
            raise ValueError("sparse search requires sparse vector values")
        if any(not math.isfinite(value) for value in self.query_vector):
            raise ValueError("query_vector values must be finite")
        if any(not math.isfinite(value) for value in self.sparse_values):
            raise ValueError("sparse vector values must be finite")
        object.__setattr__(self, "query_vector", tuple(float(value) for value in self.query_vector))
        object.__setattr__(self, "sparse_indices", tuple(self.sparse_indices))
        object.__setattr__(
            self, "sparse_values", tuple(float(value) for value in self.sparse_values)
        )
        object.__setattr__(self, "filters", freeze_metadata(self.filters))


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    score: float
    payload: Mapping[str, JSONValue]

    def __post_init__(self) -> None:
        require_non_empty(self.chunk_id, "chunk_id")
        object.__setattr__(self, "payload", freeze_metadata(self.payload))


@dataclass(slots=True)
class RetrievalCandidate:
    chunk: Chunk
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    final_score: float | None = None
    rank: int | None = None
    origins: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.rank is not None:
            require_positive(self.rank, "rank")
        self.origins = list(dict.fromkeys(self.origins))


@dataclass(frozen=True, slots=True)
class Citation:
    citation_id: str
    chunk_ids: Sequence[str]
    source_uri: str
    page_start: int | None = None
    page_end: int | None = None
    title: str | None = None

    def __post_init__(self) -> None:
        require_non_empty(self.citation_id, "citation_id")
        require_non_empty(self.source_uri, "source_uri")
        if not self.chunk_ids:
            raise ValueError("chunk_ids must not be empty")
        object.__setattr__(self, "chunk_ids", tuple(self.chunk_ids))


@dataclass(frozen=True, slots=True)
class ResultGroup:
    group_id: str
    hits: Sequence[RetrievalCandidate]

    def __post_init__(self) -> None:
        require_non_empty(self.group_id, "group_id")
        object.__setattr__(self, "hits", tuple(self.hits))


@dataclass(frozen=True, slots=True)
class OrganizedResult:
    hits: Sequence[RetrievalCandidate]
    context: str | None = None
    citations: Sequence[Citation] = ()
    warnings: Sequence[str] = ()
    groups: Sequence[ResultGroup] = ()
    debug: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "hits", tuple(self.hits))
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "debug", freeze_metadata(self.debug))


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    query: str
    processed_query: str
    hits: Sequence[RetrievalCandidate]
    index_version: str
    embedding_fingerprint: str
    expanded_queries: Sequence[str] = ()
    context: str | None = None
    citations: Sequence[Citation] = ()
    timings_ms: Mapping[str, float] = field(default_factory=dict)
    warnings: Sequence[str] = ()
    groups: Sequence[ResultGroup] = ()
    debug: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.query, "query")
        require_non_empty(self.processed_query, "processed_query")
        require_non_empty(self.index_version, "index_version")
        require_non_empty(self.embedding_fingerprint, "embedding_fingerprint")
        if any(value < 0 for value in self.timings_ms.values()):
            raise ValueError("timings_ms values must be non-negative")
        object.__setattr__(self, "hits", tuple(self.hits))
        object.__setattr__(self, "expanded_queries", tuple(self.expanded_queries))
        object.__setattr__(self, "citations", tuple(self.citations))
        object.__setattr__(self, "timings_ms", freeze_metadata(self.timings_ms))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        object.__setattr__(self, "groups", tuple(self.groups))
        object.__setattr__(self, "debug", freeze_metadata(self.debug))
