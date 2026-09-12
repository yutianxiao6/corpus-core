from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from corpuscore.chunkers import RecursiveChunker, SemanticChunker, SyntaxChunker
from corpuscore.config.models import CorpusConfig, SemanticChunkProfile
from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    ParsedDocument,
    SourceDescriptor,
)
from corpuscore.ingestion import IngestionService


def parsed_document(blocks: list[ContentBlock], *, language: str | None = None) -> ParsedDocument:
    metadata = {"language": language} if language else {}
    return ParsedDocument(
        document_id="document-v1",
        source=SourceDescriptor(
            source_id="source",
            uri="file:///module.py" if language else "file:///manual.txt",
            relative_path="module.py" if language else "manual.txt",
            content_hash="hash",
        ),
        blocks=blocks,
        parser_name="fixture",
        parser_version="1",
        metadata=metadata,
    )


class SyntaxChunkingTests(unittest.TestCase):
    def test_python_ast_keeps_top_level_symbols_as_boundaries(self) -> None:
        code = (
            "import os\n\n"
            "@decorator\n"
            "def first():\n"
            "    return os.name\n\n"
            "class Worker:\n"
            "    def run(self):\n"
            "        return 1\n"
        )
        document = parsed_document(
            [ContentBlock(ContentBlockType.CODE, code, char_start=0, char_end=len(code))],
            language="python",
        )
        chunks = SyntaxChunker(
            RecursiveChunker(chunk_size=40, chunk_overlap=0), max_chunk_size=50
        ).split(document)

        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].metadata["syntax_symbols"], ("<module>",))
        self.assertIn("@decorator", chunks[1].content)
        self.assertEqual(chunks[1].metadata["syntax_symbols"], ("first",))
        self.assertEqual(chunks[2].metadata["syntax_node_types"], ("ClassDef",))
        self.assertLess(chunks[1].char_start, chunks[1].char_end)

    def test_invalid_python_uses_configured_fallback(self) -> None:
        code = "def broken(:\n    pass"
        document = parsed_document([ContentBlock(ContentBlockType.CODE, code)], language="python")
        chunks = SyntaxChunker(RecursiveChunker(chunk_size=30, chunk_overlap=0)).split(document)

        self.assertTrue(chunks)
        self.assertTrue(chunks[0].metadata["syntax_fallback"])
        self.assertEqual(chunks[0].metadata["syntax_fallback_reason"], "invalid_python_syntax")

    def test_ingestion_discovers_code_and_selects_syntax_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "service.py"
            path.write_text("def answer():\n    return 42\n", encoding="utf-8")
            report = IngestionService(CorpusConfig()).preview([path])

        self.assertEqual(len(report.items), 1)
        item = report.items[0]
        self.assertEqual(item.document.parser_name, "code:python")
        self.assertEqual(item.chunks[0].metadata["chunker"], "syntax")
        self.assertEqual(item.chunks[0].metadata["syntax_symbols"], ("answer",))


class SemanticChunkingTests(unittest.TestCase):
    def test_low_adjacent_similarity_starts_new_chunk(self) -> None:
        blocks = [
            ContentBlock(ContentBlockType.PARAGRAPH, "安装服务器"),
            ContentBlock(ContentBlockType.PARAGRAPH, "配置服务器端口"),
            ContentBlock(ContentBlockType.PARAGRAPH, "财务报销规则"),
        ]
        vectors = {
            "安装服务器": (1.0, 0.0),
            "配置服务器端口": (0.9, 0.1),
            "财务报销规则": (0.0, 1.0),
        }
        chunker = SemanticChunker(
            lambda texts: tuple(vectors[text] for text in texts),
            similarity_threshold=0.5,
            minimum_chunk_size=1,
            maximum_chunk_size=100,
        )
        chunks = chunker.split(parsed_document(blocks))

        self.assertEqual(len(chunks), 2)
        self.assertIn("配置服务器端口", chunks[0].content)
        self.assertEqual(chunks[1].content, "财务报销规则")
        self.assertEqual(chunks[0].metadata["semantic_unit_count"], 2)

    def test_semantic_profile_validates_size_range(self) -> None:
        with self.assertRaises(ValueError):
            SemanticChunkProfile(type="semantic", minimum_chunk_size=200, maximum_chunk_size=100)

    def test_ingestion_uses_document_embeddings_for_semantic_profile(self) -> None:
        class EmbeddingFixture:
            def embed_documents(self, texts):  # type: ignore[no-untyped-def]
                return tuple((1.0, 0.0) if "服务器" in text else (0.0, 1.0) for text in texts)

        config = CorpusConfig(
            chunk_profiles={
                "default": SemanticChunkProfile(
                    type="semantic",
                    similarity_threshold=0.5,
                    minimum_chunk_size=1,
                    maximum_chunk_size=100,
                )
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manual.txt"
            path.write_text("安装服务器\n\n配置服务器\n\n财务规则", encoding="utf-8")
            service = IngestionService(config, embedding=EmbeddingFixture())  # type: ignore[arg-type]
            report = service.preview([path])

        self.assertEqual(len(report.items), 1)
        self.assertEqual(len(report.items[0].chunks), 2)
        self.assertEqual(report.items[0].chunks[0].metadata["chunker"], "semantic")


if __name__ == "__main__":
    unittest.main()
