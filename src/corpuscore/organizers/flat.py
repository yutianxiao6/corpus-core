"""Budget-aware flat candidate organizer."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from corpuscore.contracts.retrieval import ContextBudget, OrganizedResult, RetrievalCandidate


class FlatOrganizer:
    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult:
        del query
        selected: list[RetrievalCandidate] = []
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
            characters = len(chunk.content)
            tokens = chunk.token_count if chunk.token_count is not None else characters
            if (
                budget.max_characters is not None
                and used_characters + characters > budget.max_characters
            ):
                continue
            if budget.max_tokens is not None and used_tokens + tokens > budget.max_tokens:
                continue
            selected.append(candidate)
            document_counts[chunk.document_id] += 1
            used_characters += characters
            used_tokens += tokens
        return OrganizedResult(hits=selected)
