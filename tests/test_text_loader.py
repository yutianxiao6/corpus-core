from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from corpuscore.contracts.documents import SourceDescriptor
from corpuscore.exceptions import DocumentLoadError
from corpuscore.loaders import TextLoader
from corpuscore.sources import FileSystemSourceProvider


class TextLoaderTests(unittest.TestCase):
    def _source_for(self, path: Path) -> SourceDescriptor:
        return next(iter(FileSystemSourceProvider([path]).discover()))

    def test_loads_mixed_utf8_text(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "mixed.txt"
            path.write_text("离线 RAG retrieval", encoding="utf-8")
            loaded = TextLoader().load(self._source_for(path))

        self.assertEqual(loaded.text, "离线 RAG retrieval")
        self.assertEqual(loaded.metadata["encoding"], "utf-8")
        self.assertIsNone(loaded.binary)

    def test_detects_utf8_bom_and_utf16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            utf8 = root / "utf8.txt"
            utf16 = root / "utf16.txt"
            utf8.write_bytes("内容".encode("utf-8-sig"))
            utf16.write_bytes("内容".encode("utf-16"))
            utf8_loaded = TextLoader().load(self._source_for(utf8))
            utf16_loaded = TextLoader().load(self._source_for(utf16))

        self.assertEqual(utf8_loaded.text, "内容")
        self.assertEqual(utf8_loaded.metadata["encoding"], "utf-8-sig")
        self.assertEqual(utf16_loaded.text, "内容")
        self.assertEqual(utf16_loaded.metadata["encoding"], "utf-16")

    def test_rejects_file_changed_after_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "changing.txt"
            path.write_text("before", encoding="utf-8")
            source = self._source_for(path)
            path.write_text("after", encoding="utf-8")
            with self.assertRaisesRegex(DocumentLoadError, "changed"):
                TextLoader().load(source)

    def test_rejects_unsupported_and_remote_sources_without_leaking_uri(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.bin"
            path.write_bytes(b"data")
            source = self._source_for(path)
            with self.assertRaisesRegex(DocumentLoadError, "does not support"):
                TextLoader().load(source)

            remote = replace(source, uri="https://example.invalid/private.txt")
            with self.assertRaises(DocumentLoadError) as raised:
                TextLoader().load(remote)
            self.assertNotIn("example.invalid", str(raised.exception))

    def test_enforces_loader_size_limit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.txt"
            path.write_text("12345", encoding="utf-8")
            source = self._source_for(path)
            with self.assertRaisesRegex(DocumentLoadError, "size limit"):
                TextLoader(max_file_size_bytes=4).load(source)


if __name__ == "__main__":
    unittest.main()
