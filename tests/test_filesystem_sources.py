from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from corpuscore.exceptions import SourceDiscoveryError
from corpuscore.sources import DiscoveryOptions, FileSystemSourceProvider


class FileSystemSourceProviderTests(unittest.TestCase):
    def test_directory_discovery_filters_hidden_ignored_and_extensions(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / ".hidden").mkdir()
            (root / ".git").mkdir()
            (root / "readme.txt").write_text("中文 English", encoding="utf-8")
            (root / "nested" / "UPPER.TXT").write_text("upper", encoding="utf-8")
            (root / "nested" / "ignore.tmp").write_text("temp", encoding="utf-8")
            (root / ".hidden" / "secret.txt").write_text("hidden", encoding="utf-8")
            (root / ".git" / "config.txt").write_text("git", encoding="utf-8")
            (root / "binary.bin").write_bytes(b"\x00")

            provider = FileSystemSourceProvider(
                [root], options=DiscoveryOptions(allowed_extensions=("txt",))
            )
            sources = list(provider.discover())

        self.assertEqual(
            [source.relative_path for source in sources], ["nested/UPPER.TXT", "readme.txt"]
        )
        self.assertEqual(sources[1].metadata["extension"], ".txt")
        self.assertEqual(len(sources[1].content_hash), 64)

    def test_source_id_is_independent_of_absolute_deployment_path(self) -> None:
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            first_root = Path(first)
            second_root = Path(second)
            (first_root / "manual.txt").write_text("version one", encoding="utf-8")
            (second_root / "manual.txt").write_text("different content", encoding="utf-8")
            first_source = next(
                iter(FileSystemSourceProvider([first_root], namespace="acme").discover())
            )
            second_source = next(
                iter(FileSystemSourceProvider([second_root], namespace="acme").discover())
            )

        self.assertEqual(first_source.source_id, second_source.source_id)
        self.assertNotEqual(first_source.content_hash, second_source.content_hash)

    def test_explicit_files_and_globs_are_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            first = root / "a.txt"
            second = root / "nested" / "b.txt"
            first.write_text("a", encoding="utf-8")
            second.write_text("b", encoding="utf-8")
            provider = FileSystemSourceProvider([first, str(root / "**" / "*.txt")], root=root)
            paths = [source.relative_path for source in provider.discover()]

        self.assertEqual(paths, ["a.txt", "nested/b.txt"])

    @unittest.skipUnless(hasattr(os, "symlink"), "symlinks are unavailable")
    def test_followed_symlink_loop_is_deduplicated_and_external_target_is_blocked(self) -> None:
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryDirectory() as external:
            root = Path(directory)
            nested = root / "nested"
            nested.mkdir()
            (nested / "inside.txt").write_text("inside", encoding="utf-8")
            outside = Path(external) / "outside.txt"
            outside.write_text("outside", encoding="utf-8")
            (nested / "loop").symlink_to(root, target_is_directory=True)
            (root / "outside.txt").symlink_to(outside)

            provider = FileSystemSourceProvider(
                [root], options=DiscoveryOptions(follow_symlinks=True)
            )
            paths = [source.relative_path for source in provider.discover()]

        self.assertEqual(paths, ["nested/inside.txt"])

    def test_non_recursive_mode_only_returns_direct_children(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "nested").mkdir()
            (root / "top.txt").write_text("top", encoding="utf-8")
            (root / "nested" / "deep.txt").write_text("deep", encoding="utf-8")
            provider = FileSystemSourceProvider([root], options=DiscoveryOptions(recursive=False))
            paths = [source.relative_path for source in provider.discover()]

        self.assertEqual(paths, ["top.txt"])

    def test_include_and_ignore_patterns(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "keep").mkdir()
            (root / "skip").mkdir()
            (root / "keep" / "one.md").write_text("one", encoding="utf-8")
            (root / "skip" / "two.md").write_text("two", encoding="utf-8")
            provider = FileSystemSourceProvider(
                [root],
                options=DiscoveryOptions(
                    include_patterns=("**/*.md",), ignore_patterns=("skip/**",)
                ),
            )
            paths = [source.relative_path for source in provider.discover()]

        self.assertEqual(paths, ["keep/one.md"])

    def test_missing_glob_and_size_limits_raise_structured_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with self.assertRaisesRegex(SourceDiscoveryError, "did not match"):
                list(FileSystemSourceProvider([str(root / "*.missing")]).discover())
            large = root / "large.txt"
            large.write_text("12345", encoding="utf-8")
            provider = FileSystemSourceProvider(
                [large], options=DiscoveryOptions(max_file_size_bytes=4)
            )
            with self.assertRaisesRegex(SourceDiscoveryError, "exceeds"):
                list(provider.discover())


if __name__ == "__main__":
    unittest.main()
