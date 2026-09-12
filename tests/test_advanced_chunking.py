from __future__ import annotations

import unittest

from corpuscore.chunkers import (
    ChunkFinalizer,
    PageAwareChunker,
    ParagraphPackingChunker,
    ParentChildChunker,
    TableRowChunker,
)
from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    ParsedDocument,
    SourceDescriptor,
)


def document(blocks: list[ContentBlock]) -> ParsedDocument:
    return ParsedDocument(
        document_id="document-version-1",
        source=SourceDescriptor(
            source_id="source",
            uri="file:///manual.pdf",
            content_hash="hash",
        ),
        title="Manual",
        blocks=blocks,
        parser_name="fixture",
        parser_version="1",
    )


class AdvancedChunkingTests(unittest.TestCase):
    def test_page_aware_never_combines_different_pages(self) -> None:
        parsed = document(
            [
                ContentBlock(ContentBlockType.PARAGRAPH, "page one " * 5, page=1),
                ContentBlock(ContentBlockType.PAGE_BREAK, "", page=1),
                ContentBlock(ContentBlockType.PARAGRAPH, "page two " * 5, page=2),
            ]
        )
        chunks = PageAwareChunker(chunk_size=30, chunk_overlap=5).split(parsed)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.page_start == chunk.page_end for chunk in chunks))
        self.assertEqual({chunk.page_start for chunk in chunks}, {1, 2})

    def test_paragraph_packing_keeps_small_paragraphs_whole(self) -> None:
        parsed = document(
            [
                ContentBlock(ContentBlockType.PARAGRAPH, "first"),
                ContentBlock(ContentBlockType.PARAGRAPH, "second"),
                ContentBlock(ContentBlockType.PARAGRAPH, "third paragraph"),
            ]
        )
        chunks = ParagraphPackingChunker(target_size=15, maximum_size=20).split(parsed)

        self.assertEqual(chunks[0].content, "first\n\nsecond")
        self.assertEqual(chunks[1].content, "third paragraph")
        self.assertTrue(all(len(chunk.content) <= 20 for chunk in chunks))

    def test_table_rows_repeat_header_and_limit_data_rows(self) -> None:
        parsed = document(
            [
                ContentBlock(
                    ContentBlockType.TABLE,
                    "name | version\nLinux | 2026\nWindows | 11\nmacOS | 15",
                    page=3,
                )
            ]
        )
        chunks = TableRowChunker(max_rows_per_chunk=2, repeat_headers=True).split(parsed)

        self.assertEqual(len(chunks), 2)
        self.assertTrue(all(chunk.content.startswith("name | version\n") for chunk in chunks))
        self.assertEqual(chunks[0].metadata["row_count"], 2)
        self.assertEqual(chunks[1].metadata["row_count"], 1)
        self.assertTrue(all(chunk.page_start == 3 for chunk in chunks))

    def test_parent_child_retains_parent_content_and_stable_parent_id(self) -> None:
        parsed = document([ContentBlock(ContentBlockType.PARAGRAPH, "Sentence one. " * 30)])
        drafts = ParentChildChunker(parent_size=120, child_size=40, child_overlap=5).split(parsed)
        chunks = ChunkFinalizer().finalize(drafts)

        self.assertGreater(len(chunks), 2)
        self.assertTrue(all(chunk.parent_id for chunk in chunks))
        parent_groups: dict[str, set[str]] = {}
        for chunk in chunks:
            assert chunk.parent_id is not None
            parent_groups.setdefault(chunk.parent_id, set()).add(
                str(chunk.metadata["parent_content"])
            )
        self.assertTrue(all(len(contents) == 1 for contents in parent_groups.values()))
        self.assertTrue(all(len(chunk.content) <= 40 for chunk in chunks))


if __name__ == "__main__":
    unittest.main()
