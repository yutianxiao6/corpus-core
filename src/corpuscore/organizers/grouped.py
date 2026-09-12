"""Group ranked results by document while retaining a flat compatibility view."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Sequence

from corpuscore.contracts.retrieval import (
    ContextBudget,
    OrganizedResult,
    ResultGroup,
    RetrievalCandidate,
)
from corpuscore.organizers.flat import FlatOrganizer


class GroupByDocumentOrganizer:
    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult:
        selected = FlatOrganizer().organize(query, candidates, budget).hits
        buckets: dict[str, list[RetrievalCandidate]] = defaultdict(list)
        for candidate in selected:
            buckets[candidate.chunk.document_id].append(candidate)
        groups = [
            ResultGroup(
                document_id,
                sorted(
                    hits,
                    key=lambda candidate: (
                        candidate.chunk.page_start
                        if candidate.chunk.page_start is not None
                        else math.inf,
                        candidate.chunk.chunk_index,
                    ),
                ),
            )
            for document_id, hits in buckets.items()
        ]
        flattened = tuple(hit for group in groups for hit in group.hits)
        return OrganizedResult(hits=flattened, groups=groups)
