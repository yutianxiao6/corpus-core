from __future__ import annotations

import unittest

from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.retrieval import ContextBudget, RetrievalCandidate
from offline_rag.organizers import ContextOrganizer, FlatOrganizer


def candidate(
    identifier: str,
    *,
    document_id: str = "doc-1",
    content: str = "evidence",
    page: int | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(
            chunk_id=identifier,
            document_id=document_id,
            content=content,
            embedding_text=content,
            source_uri=f"file:///{document_id}.md",
            chunk_index=0,
            heading_path=("安装", "Linux"),
            page_start=page,
            page_end=page,
            token_count=len(content),
        ),
        dense_score=0.9,
        final_score=0.9,
        rank=1,
        origins=["dense"],
    )


class OrganizerTests(unittest.TestCase):
    def test_flat_organizer_applies_document_and_size_budget(self) -> None:
        candidates = [
            candidate("one", content="12345"),
            candidate("two", content="12345"),
            candidate("three", document_id="doc-2", content="12345"),
        ]
        result = FlatOrganizer().organize(
            "query",
            candidates,
            ContextBudget(max_characters=10, maximum_chunks_per_document=1),
        )
        self.assertEqual([item.chunk.chunk_id for item in result.hits], ["one", "three"])

    def test_context_has_stable_citations_and_exact_budget_behavior(self) -> None:
        candidates = [candidate("one", content="first evidence", page=2)]
        organizer = ContextOrganizer(token_counter=lambda text: len(text.split()))
        roomy = organizer.organize("query", candidates, ContextBudget(max_tokens=50))

        self.assertIsNotNone(roomy.context)
        assert roomy.context is not None
        self.assertIn("[证据 1]", roomy.context)
        self.assertIn("第 2 页", roomy.context)
        self.assertIn("章节：安装 > Linux", roomy.context)
        self.assertEqual(roomy.citations[0].citation_id, "evidence-1")
        self.assertEqual(roomy.citations[0].chunk_ids, ("one",))

        constrained = organizer.organize("query", candidates, ContextBudget(max_tokens=1))
        self.assertIsNone(constrained.context)
        self.assertEqual(constrained.hits, ())
        self.assertIn("token budget", constrained.warnings[0])


if __name__ == "__main__":
    unittest.main()
