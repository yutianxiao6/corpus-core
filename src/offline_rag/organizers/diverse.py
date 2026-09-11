"""Round-robin organization across source documents."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence

from offline_rag.contracts.retrieval import ContextBudget, OrganizedResult, RetrievalCandidate
from offline_rag.organizers.flat import FlatOrganizer


class DiverseOrganizer:
    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult:
        buckets: dict[str, list[RetrievalCandidate]] = defaultdict(list)
        for candidate in candidates:
            buckets[candidate.chunk.document_id].append(candidate)
        interleaved: list[RetrievalCandidate] = []
        depth = 0
        while True:
            added = False
            for bucket in buckets.values():
                if depth < len(bucket):
                    interleaved.append(bucket[depth])
                    added = True
            if not added:
                break
            depth += 1
        return FlatOrganizer().organize(query, interleaved, budget)
