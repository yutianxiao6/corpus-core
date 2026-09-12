"""Context organizers for adjacent chunks and parent-child evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace

from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.retrieval import (
    Citation,
    ContextBudget,
    OrganizedResult,
    RetrievalCandidate,
)
from corpuscore.organizers.context import ContextOrganizer


class MergeNeighborsOrganizer:
    def __init__(self, *, token_counter: Callable[[str], int] = len) -> None:
        self._context = ContextOrganizer(token_counter=token_counter)

    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult:
        groups: list[list[RetrievalCandidate]] = []
        for candidate in candidates:
            if groups and _are_adjacent(groups[-1][-1], candidate):
                groups[-1].append(candidate)
            else:
                groups.append([candidate])
        merged = tuple(
            _aggregate(group, kind="merged") if len(group) > 1 else group[0] for group in groups
        )
        return _restore_evidence_citations(self._context.organize(query, merged, budget))


class ParentOrganizer:
    def __init__(self, *, token_counter: Callable[[str], int] = len) -> None:
        self._context = ContextOrganizer(token_counter=token_counter)

    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult:
        buckets: dict[str, list[RetrievalCandidate]] = {}
        for candidate in candidates:
            key = candidate.chunk.parent_id or candidate.chunk.chunk_id
            buckets.setdefault(key, []).append(candidate)
        parents = tuple(
            _aggregate(group, kind="parent", chunk_id=key) if group[0].chunk.parent_id else group[0]
            for key, group in buckets.items()
        )
        return _restore_evidence_citations(self._context.organize(query, parents, budget))


def _are_adjacent(left: RetrievalCandidate, right: RetrievalCandidate) -> bool:
    return (
        left.chunk.document_id == right.chunk.document_id
        and left.chunk.source_uri == right.chunk.source_uri
        and left.chunk.next_id == right.chunk.chunk_id
        and right.chunk.previous_id == left.chunk.chunk_id
    )


def _aggregate(
    candidates: Sequence[RetrievalCandidate],
    *,
    kind: str,
    chunk_id: str | None = None,
) -> RetrievalCandidate:
    first = candidates[0]
    last = candidates[-1]
    chunks = [candidate.chunk for candidate in candidates]
    evidence_ids = tuple(chunk.chunk_id for chunk in chunks)
    if kind == "parent":
        parent_content = first.chunk.metadata.get("parent_content")
        content = parent_content if isinstance(parent_content, str) else first.chunk.content
    else:
        content = "\n\n".join(chunk.content for chunk in chunks)
    identifier = (
        chunk_id or "merged-" + hashlib.sha256("\0".join(evidence_ids).encode()).hexdigest()
    )
    metadata = dict(first.chunk.metadata)
    metadata.update({"organized_as": kind, "evidence_chunk_ids": evidence_ids})
    chunk = Chunk(
        chunk_id=identifier,
        document_id=first.chunk.document_id,
        content=content,
        embedding_text=content,
        source_uri=first.chunk.source_uri,
        chunk_index=min(item.chunk_index for item in chunks),
        title=first.chunk.title,
        heading_path=first.chunk.heading_path,
        page_start=_minimum(item.page_start for item in chunks),
        page_end=_maximum(item.page_end for item in chunks),
        char_start=_minimum(item.char_start for item in chunks),
        char_end=_maximum(item.char_end for item in chunks),
        parent_id=first.chunk.parent_id if kind == "merged" else None,
        previous_id=first.chunk.previous_id,
        next_id=last.chunk.next_id,
        token_count=(
            sum(item.token_count for item in chunks if item.token_count is not None)
            if all(item.token_count is not None for item in chunks)
            else None
        ),
        metadata=metadata,
    )
    return RetrievalCandidate(
        chunk=chunk,
        dense_score=_best(candidate.dense_score for candidate in candidates),
        sparse_score=_best(candidate.sparse_score for candidate in candidates),
        fusion_score=_best(candidate.fusion_score for candidate in candidates),
        rerank_score=_best(candidate.rerank_score for candidate in candidates),
        final_score=_best(candidate.final_score for candidate in candidates),
        rank=first.rank,
        origins=list(
            dict.fromkeys(origin for candidate in candidates for origin in candidate.origins)
        ),
    )


def _restore_evidence_citations(result: OrganizedResult) -> OrganizedResult:
    citations: list[Citation] = []
    for candidate, citation in zip(result.hits, result.citations, strict=True):
        value = candidate.chunk.metadata.get("evidence_chunk_ids")
        chunk_ids = (
            tuple(item for item in value if isinstance(item, str))
            if isinstance(value, tuple)
            else citation.chunk_ids
        )
        citations.append(replace(citation, chunk_ids=chunk_ids))
    return replace(result, citations=tuple(citations))


def _minimum(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return min(present, default=None)


def _maximum(values: Iterable[int | None]) -> int | None:
    present = [value for value in values if value is not None]
    return max(present, default=None)


def _best(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return max(present, default=None)
