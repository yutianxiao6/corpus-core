"""Expose candidate scores and provenance for retrieval diagnostics."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

from corpuscore.contracts.retrieval import ContextBudget, OrganizedResult, RetrievalCandidate


class DebugOrganizer:
    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult:
        document_counts: Counter[str] = Counter()
        used_characters = 0
        used_tokens = 0
        selected: list[RetrievalCandidate] = []
        decisions: list[str] = []
        for candidate in candidates:
            chunk = candidate.chunk
            characters = len(chunk.content)
            tokens = chunk.token_count if chunk.token_count is not None else characters
            if (
                budget.maximum_chunks_per_document is not None
                and document_counts[chunk.document_id] >= budget.maximum_chunks_per_document
            ):
                decisions.append("document_limit")
            elif (
                budget.max_characters is not None
                and used_characters + characters > budget.max_characters
            ):
                decisions.append("character_budget")
            elif budget.max_tokens is not None and used_tokens + tokens > budget.max_tokens:
                decisions.append("token_budget")
            else:
                decisions.append("selected")
                selected.append(candidate)
                document_counts[chunk.document_id] += 1
                used_characters += characters
                used_tokens += tokens
        return OrganizedResult(
            hits=selected,
            debug={
                "query": query,
                "candidate_count": len(candidates),
                "selected_count": len(selected),
                "candidates": tuple(
                    {
                        "chunk_id": candidate.chunk.chunk_id,
                        "document_id": candidate.chunk.document_id,
                        "rank": candidate.rank,
                        "dense_score": candidate.dense_score,
                        "sparse_score": candidate.sparse_score,
                        "fusion_score": candidate.fusion_score,
                        "rerank_score": candidate.rerank_score,
                        "final_score": candidate.final_score,
                        "origins": tuple(candidate.origins),
                        "selected": decision == "selected",
                        "decision": decision,
                    }
                    for candidate, decision in zip(candidates, decisions, strict=True)
                ),
            },
        )
