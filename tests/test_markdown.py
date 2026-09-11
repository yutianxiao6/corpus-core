from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from offline_rag.contracts.documents import ContentBlockType, LoadedContent
from offline_rag.exceptions import DocumentParseError
from offline_rag.loaders import MarkdownLoader
from offline_rag.parsers import MarkdownParser
from offline_rag.sources import FileSystemSourceProvider


class MarkdownTests(unittest.TestCase):
    def _load(self, path: Path) -> LoadedContent:
        source = next(iter(FileSystemSourceProvider([path]).discover()))
        return MarkdownLoader().load(source)

    def test_loader_supports_both_markdown_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            md = root / "one.md"
            markdown = root / "two.MARKDOWN"
            md.write_text("# One", encoding="utf-8")
            markdown.write_text("# Two", encoding="utf-8")
            loader = MarkdownLoader()

            self.assertTrue(loader.supports(self._load(md).source))
            self.assertTrue(loader.supports(self._load(markdown).source))

    def test_parser_preserves_heading_paragraph_list_and_code_structure(self) -> None:
        content = (
            "# 安装 Guide\n\n"
            "第一段 with English.\ncontinued line\n\n"
            "- 安装 Python\n"
            "1. Run command\n\n"
            "```python\n"
            "def run():\n"
            "    return '# not a heading'\n"
            "```\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "guide.md"
            path.write_text(content, encoding="utf-8")
            document = MarkdownParser().parse(self._load(path))

        self.assertEqual(document.title, "安装 Guide")
        self.assertEqual(
            [block.block_type for block in document.blocks],
            [
                ContentBlockType.HEADING,
                ContentBlockType.PARAGRAPH,
                ContentBlockType.LIST_ITEM,
                ContentBlockType.LIST_ITEM,
                ContentBlockType.CODE,
            ],
        )
        self.assertEqual(document.blocks[0].heading_level, 1)
        self.assertEqual(document.blocks[-1].metadata["language"], "python")
        self.assertIn("    return", document.blocks[-1].content)
        self.assertEqual(document.metadata["normalized"], True)

    def test_source_offsets_reference_original_normalized_text(self) -> None:
        content = "## Section\n\nParagraph text\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "offsets.md"
            path.write_text(content, encoding="utf-8")
            document = MarkdownParser().parse(self._load(path))

        for block in document.blocks:
            assert block.char_start is not None
            assert block.char_end is not None
            self.assertEqual(content[block.char_start : block.char_end], block.content)

    def test_document_id_is_deterministic_and_content_versioned(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "versioned.md"
            path.write_text("# V1", encoding="utf-8")
            first_loaded = self._load(path)
            first = MarkdownParser().parse(first_loaded)
            repeat = MarkdownParser().parse(first_loaded)
            path.write_text("# V2", encoding="utf-8")
            second = MarkdownParser().parse(self._load(path))

        self.assertEqual(first.document_id, repeat.document_id)
        self.assertNotEqual(first.document_id, second.document_id)

    def test_empty_markdown_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty.md"
            path.write_text(" \n\n", encoding="utf-8")
            with self.assertRaisesRegex(DocumentParseError, "empty"):
                MarkdownParser().parse(self._load(path))

    def test_empty_fence_is_not_emitted_as_an_invalid_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "empty-code.md"
            path.write_text("# Valid\n\n```\n```\n", encoding="utf-8")
            document = MarkdownParser().parse(self._load(path))

        self.assertEqual(len(document.blocks), 1)
        self.assertEqual(document.blocks[0].block_type, ContentBlockType.HEADING)


if __name__ == "__main__":
    unittest.main()
