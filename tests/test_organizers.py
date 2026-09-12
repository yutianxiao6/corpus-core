from __future__ import annotations

import unittest

from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.retrieval import ContextBudget, RetrievalCandidate
from corpuscore.organizers import (
    ContextOrganizer,
    DebugOrganizer,
    DiverseOrganizer,
    FlatOrganizer,
    GroupByDocumentOrganizer,
    MergeNeighborsOrganizer,
    ParentOrganizer,
)


def candidate(
    identifier: str,
    *,
    document_id: str = "doc-1",
    content: str = "evidence",
    page: int | None = None,
    chunk_index: int = 0,
    previous_id: str | None = None,
    next_id: str | None = None,
    parent_id: str | None = None,
    parent_content: str | None = None,
) -> RetrievalCandidate:
    return RetrievalCandidate(
        chunk=Chunk(
            chunk_id=identifier,
            document_id=document_id,
            content=content,
            embedding_text=content,
            source_uri=f"file:///{document_id}.md",
            chunk_index=chunk_index,
            heading_path=("安装", "Linux"),
            page_start=page,
            page_end=page,
            previous_id=previous_id,
            next_id=next_id,
            parent_id=parent_id,
            token_count=len(content),
            metadata=({"parent_content": parent_content} if parent_content is not None else {}),
        ),
        dense_score=0.9,
        final_score=0.9,
        rank=1,
        origins=["dense"],
    )


class OrganizerTests(unittest.TestCase):
    def test_grouped_organizer_orders_documents_by_best_hit_and_chunks_by_location(self) -> None:
        candidates = [
            candidate("a-page-3", document_id="a", page=3, chunk_index=2),
            candidate("b-page-2", document_id="b", page=2, chunk_index=1),
            candidate("a-page-1", document_id="a", page=1, chunk_index=0),
        ]
        result = GroupByDocumentOrganizer().organize(
            "query", candidates, ContextBudget(max_characters=1000)
        )
        self.assertEqual([group.group_id for group in result.groups], ["a", "b"])
        self.assertEqual(
            [item.chunk.chunk_id for item in result.groups[0].hits], ["a-page-1", "a-page-3"]
        )

    def test_merge_neighbors_preserves_all_evidence_ids_and_page_range(self) -> None:
        candidates = [
            candidate("one", content="first", page=1, next_id="two"),
            candidate("two", content="second", page=2, previous_id="one"),
        ]
        result = MergeNeighborsOrganizer().organize(
            "query", candidates, ContextBudget(max_characters=1000)
        )
        self.assertEqual(len(result.hits), 1)
        self.assertEqual(result.hits[0].chunk.content, "first\n\nsecond")
        self.assertEqual(result.hits[0].chunk.page_start, 1)
        self.assertEqual(result.hits[0].chunk.page_end, 2)
        self.assertEqual(result.citations[0].chunk_ids, ("one", "two"))

    def test_parent_organizer_collapses_children_and_retains_child_citations(self) -> None:
        candidates = [
            candidate("child-a", parent_id="parent", parent_content="complete parent"),
            candidate("child-b", parent_id="parent", parent_content="complete parent"),
        ]
        result = ParentOrganizer().organize("query", candidates, ContextBudget(max_characters=1000))
        self.assertEqual([item.chunk.chunk_id for item in result.hits], ["parent"])
        self.assertEqual(result.hits[0].chunk.content, "complete parent")
        self.assertEqual(result.citations[0].chunk_ids, ("child-a", "child-b"))

    def test_diverse_organizer_round_robins_documents(self) -> None:
        candidates = [
            candidate("a1", document_id="a"),
            candidate("a2", document_id="a"),
            candidate("b1", document_id="b"),
            candidate("b2", document_id="b"),
        ]
        result = DiverseOrganizer().organize(
            "query", candidates, ContextBudget(max_characters=1000)
        )
        self.assertEqual([item.chunk.chunk_id for item in result.hits], ["a1", "b1", "a2", "b2"])

    def test_debug_organizer_exposes_scores_and_selection(self) -> None:
        candidates = [candidate("one", content="12345"), candidate("two", content="12345")]
        result = DebugOrganizer().organize("diagnose", candidates, ContextBudget(max_characters=5))
        self.assertEqual(result.debug["query"], "diagnose")
        self.assertEqual(result.debug["candidate_count"], 2)
        details = result.debug["candidates"]
        self.assertIsInstance(details, tuple)
        assert isinstance(details, tuple)
        self.assertEqual([item["selected"] for item in details], [True, False])
        self.assertEqual([item["decision"] for item in details], ["selected", "character_budget"])

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
