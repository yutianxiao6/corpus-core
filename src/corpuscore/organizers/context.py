"""Produce stable, cited context blocks for a downstream answer model."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence

from corpuscore.contracts.retrieval import (
    Citation,
    ContextBudget,
    OrganizedResult,
    RetrievalCandidate,
)


class ContextOrganizer:
    def __init__(self, *, token_counter: Callable[[str], int] = len) -> None:
        self._token_counter = token_counter

    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult:
        del query
        sections: list[str] = []
        selected: list[RetrievalCandidate] = []
        citations: list[Citation] = []
        warnings: list[str] = []
        document_counts: Counter[str] = Counter()
        used_characters = 0
        used_tokens = 0
        for candidate in candidates:
            chunk = candidate.chunk
            maximum_per_document = budget.maximum_chunks_per_document
            if (
                maximum_per_document is not None
                and document_counts[chunk.document_id] >= maximum_per_document
            ):
                continue
            citation_number = len(citations) + 1
            section = self._section(citation_number, candidate)
            characters = len(section) + (2 if sections else 0)
            tokens = self._token_counter(section)
            if (
                budget.max_characters is not None
                and used_characters + characters > budget.max_characters
            ):
                warnings.append(
                    f"chunk {chunk.chunk_id} skipped because the character budget is full"
                )
                continue
            if budget.max_tokens is not None and used_tokens + tokens > budget.max_tokens:
                warnings.append(f"chunk {chunk.chunk_id} skipped because the token budget is full")
                continue
            citation_id = f"evidence-{citation_number}"
            sections.append(section)
            selected.append(candidate)
            citations.append(
                Citation(
                    citation_id=citation_id,
                    chunk_ids=(chunk.chunk_id,),
                    source_uri=chunk.source_uri,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    title=chunk.title,
                )
            )
            document_counts[chunk.document_id] += 1
            used_characters += characters
            used_tokens += tokens
        return OrganizedResult(
            hits=selected,
            context="\n\n".join(sections) if sections else None,
            citations=citations,
            warnings=tuple(dict.fromkeys(warnings)),
        )

    @staticmethod
    def _section(number: int, candidate: RetrievalCandidate) -> str:
        chunk = candidate.chunk
        source = f"来源：{chunk.source_uri}"
        if chunk.page_start is not None:
            page = str(chunk.page_start)
            if chunk.page_end is not None and chunk.page_end != chunk.page_start:
                page = f"{page}-{chunk.page_end}"
            source = f"{source}，第 {page} 页"
        lines = [f"[证据 {number}]", source]
        if chunk.heading_path:
            lines.append(f"章节：{' > '.join(chunk.heading_path)}")
        lines.append(f"内容：{chunk.content}")
        return "\n".join(lines)
