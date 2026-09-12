from __future__ import annotations

import unittest

from corpuscore.chunkers import (
    ChunkDeduplicator,
    ChunkFinalizer,
    ChunkSizeProcessor,
    HeadingContextInjector,
    HeadingRecursiveChunker,
    RecursiveChunker,
)
from corpuscore.contracts.chunks import ChunkDraft
from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    ParsedDocument,
    SourceDescriptor,
)
from corpuscore.exceptions import ChunkingError


def make_document(blocks: list[ContentBlock], *, document_id: str = "doc-v1") -> ParsedDocument:
    source = SourceDescriptor(
        source_id="source-1",
        uri="file:///manual.md",
        relative_path="manual.md",
        content_hash="content-v1",
    )
    return ParsedDocument(
        document_id=document_id,
        source=source,
        title="产品手册",
        blocks=blocks,
        parser_name="markdown",
        parser_version="1",
    )


class RecursiveChunkerTests(unittest.TestCase):
    def test_recursive_chunks_respect_size_and_preserve_metadata(self) -> None:
        document = make_document(
            [
                ContentBlock(
                    ContentBlockType.HEADING, "安装", heading_level=1, char_start=0, char_end=2
                ),
                ContentBlock(
                    ContentBlockType.PARAGRAPH,
                    "第一句话很重要。第二句话说明安装过程。Third sentence explains setup.",
                    char_start=4,
                    char_end=52,
                ),
            ]
        )
        chunks = RecursiveChunker(chunk_size=24, chunk_overlap=4).split(document)

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(len(chunk.content) <= 24 for chunk in chunks))
        self.assertTrue(all(chunk.document_id == document.document_id for chunk in chunks))
        self.assertTrue(all(chunk.heading_path == ("安装",) for chunk in chunks))
        self.assertTrue(all(chunk.metadata["chunker"] == "recursive" for chunk in chunks))

    def test_heading_recursive_never_combines_different_sections(self) -> None:
        document = make_document(
            [
                ContentBlock(ContentBlockType.HEADING, "安装", heading_level=1),
                ContentBlock(ContentBlockType.PARAGRAPH, "安装说明。" * 10),
                ContentBlock(ContentBlockType.HEADING, "升级", heading_level=1),
                ContentBlock(ContentBlockType.PARAGRAPH, "升级说明。" * 10),
            ]
        )
        chunks = HeadingRecursiveChunker(max_chunk_size=30, chunk_overlap=5).split(document)

        paths = {chunk.heading_path for chunk in chunks}
        self.assertEqual(paths, {("安装",), ("升级",)})
        self.assertFalse(
            any("安装说明" in chunk.content and "升级说明" in chunk.content for chunk in chunks)
        )
        self.assertTrue(all(chunk.metadata["chunker"] == "heading_recursive" for chunk in chunks))

    def test_nested_heading_path_is_propagated(self) -> None:
        document = make_document(
            [
                ContentBlock(ContentBlockType.HEADING, "系统", heading_level=1),
                ContentBlock(ContentBlockType.PARAGRAPH, "overview"),
                ContentBlock(ContentBlockType.HEADING, "Linux", heading_level=2),
                ContentBlock(ContentBlockType.PARAGRAPH, "requirements"),
            ]
        )
        chunks = HeadingRecursiveChunker(max_chunk_size=100, chunk_overlap=0).split(document)

        self.assertEqual(chunks[0].heading_path, ("系统",))
        self.assertEqual(chunks[1].heading_path, ("系统", "Linux"))


class ChunkProcessorTests(unittest.TestCase):
    def _draft(self, content: str, **kwargs: object) -> ChunkDraft:
        return ChunkDraft(
            document_id=str(kwargs.pop("document_id", "doc-v1")),
            content=content,
            source_uri="file:///manual.md",
            **kwargs,  # type: ignore[arg-type]
        )

    def test_short_compatible_chunks_merge(self) -> None:
        chunks = [
            self._draft("short", heading_path=("A",), char_start=0, char_end=5),
            self._draft("also short", heading_path=("A",), char_start=7, char_end=17),
            self._draft("different", heading_path=("B",), char_start=19, char_end=28),
        ]
        result = ChunkSizeProcessor(minimum_size=15, maximum_size=30, count_tokens=True).process(
            chunks
        )

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].content, "short\n\nalso short")
        self.assertEqual(result[0].metadata["merged_short_chunks"], 1)
        self.assertEqual(result[0].token_count, len(result[0].content))

    def test_overlong_chunk_is_forced_below_limit(self) -> None:
        chunk = self._draft("无分隔内容" * 20, char_start=10, char_end=110)
        result = ChunkSizeProcessor(minimum_size=0, maximum_size=20).process([chunk])

        self.assertGreater(len(result), 1)
        self.assertTrue(all(len(item.content) <= 20 for item in result))
        self.assertTrue(all(item.metadata["forced_split"] is True for item in result))

    def test_heading_context_only_changes_embedding_text(self) -> None:
        original = self._draft("正文内容", title="手册", heading_path=("手册", "安装"))
        result = HeadingContextInjector().process([original])[0]

        self.assertEqual(result.content, "正文内容")
        self.assertEqual(result.embedding_text, "手册\n\n安装\n\n正文内容")

    def test_exact_and_optional_near_duplicates_are_order_preserving(self) -> None:
        chunks = [
            self._draft("Install  the package version 1.0"),
            self._draft("install the package version 1.0"),
            self._draft("Install the package version 1.0 now"),
            self._draft("Install the package version 2.0"),
        ]
        exact = ChunkDeduplicator().process(chunks)
        approximate = ChunkDeduplicator(approximate_threshold=0.85).process(chunks)

        self.assertEqual(
            [item.content for item in exact],
            [chunks[0].content, chunks[2].content, chunks[3].content],
        )
        self.assertEqual(approximate[0].content, chunks[0].content)
        self.assertLess(len(approximate), len(exact))
        self.assertIn(chunks[3], approximate)


class ChunkFinalizerTests(unittest.TestCase):
    def test_ids_are_deterministic_and_neighbors_are_bidirectional(self) -> None:
        drafts = [
            ChunkDraft(
                document_id="doc-v1",
                content=f"chunk {index}",
                source_uri="file:///manual.md",
                parent_key="section-a",
            )
            for index in range(3)
        ]
        first = ChunkFinalizer().finalize(drafts)
        second = ChunkFinalizer().finalize(drafts)

        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])
        self.assertIsNone(first[0].previous_id)
        self.assertEqual(first[0].next_id, first[1].chunk_id)
        self.assertEqual(first[1].previous_id, first[0].chunk_id)
        self.assertEqual(first[1].next_id, first[2].chunk_id)
        self.assertIsNone(first[2].next_id)
        self.assertEqual(len({chunk.parent_id for chunk in first}), 1)
        self.assertEqual([chunk.chunk_index for chunk in first], [0, 1, 2])

    def test_mixed_documents_are_rejected(self) -> None:
        drafts = [
            ChunkDraft(document_id="one", content="a", source_uri="one.md"),
            ChunkDraft(document_id="two", content="b", source_uri="two.md"),
        ]
        with self.assertRaisesRegex(ChunkingError, "exactly one document"):
            ChunkFinalizer().finalize(drafts)


if __name__ == "__main__":
    unittest.main()
