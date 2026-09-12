from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from offline_rag.config.models import (
    HeadingRecursiveChunkProfile,
    RagConfig,
    RecursiveChunkProfile,
    RouteMatch,
    RoutingRule,
)
from offline_rag.contracts.documents import SourceDescriptor
from offline_rag.exceptions import ConfigurationError
from offline_rag.ingestion import IngestionService
from offline_rag.sources import DiscoveryOptions, ManifestSourceProvider, apply_sidecar
from offline_rag.sources.filesystem import FileSystemSourceProvider


def profiles():  # type: ignore[no-untyped-def]
    return {
        "default": RecursiveChunkProfile(
            type="recursive", chunk_size=100, chunk_overlap=10, minimum_size=0
        ),
        "markdown_heading": HeadingRecursiveChunkProfile(
            type="heading_recursive", max_chunk_size=100, chunk_overlap=10
        ),
    }


class ManifestTests(unittest.TestCase):
    def test_manifest_paths_metadata_profile_and_namespace(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs" / "one.txt").write_text("one", encoding="utf-8")
            (root / "docs" / "two.txt").write_text("two", encoding="utf-8")
            manifest = root / "import.yaml"
            manifest.write_text(
                """
version: 1
namespace: acme
documents:
  - path: docs/one.txt
    profile: markdown_heading
    metadata:
      department: engineering
  - path: docs/two.txt
    metadata:
      department: support
""",
                encoding="utf-8",
            )
            sources = list(
                ManifestSourceProvider(
                    manifest, options=DiscoveryOptions(allowed_extensions=(".txt",))
                ).discover()
            )

        self.assertEqual([item.relative_path for item in sources], ["docs/one.txt", "docs/two.txt"])
        self.assertEqual(sources[0].metadata["rag_profile"], "markdown_heading")
        self.assertEqual(sources[0].metadata["department"], "engineering")
        self.assertNotEqual(sources[0].source_id, sources[1].source_id)

    def test_unknown_manifest_field_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            manifest = Path(directory) / "bad.yaml"
            manifest.write_text(
                "version: 1\ndocuments: []\nembedding: remote-model\n", encoding="utf-8"
            )
            with self.assertRaises(ConfigurationError):
                ManifestSourceProvider(manifest)

    def test_duplicate_manifest_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "one.txt").write_text("one", encoding="utf-8")
            manifest = root / "duplicate.yaml"
            manifest.write_text(
                "documents:\n  - path: one.txt\n  - path: one.txt\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(ConfigurationError, "more than once"):
                list(ManifestSourceProvider(manifest).discover())


class SidecarAndRoutingTests(unittest.TestCase):
    def test_structured_formats_use_specialized_default_profiles(self) -> None:
        service = IngestionService(RagConfig())
        expectations = {
            "report.xlsx": "table_rows",
            "data.csv": "table_rows",
            "slides.pptx": "page_aware",
            "manual.pdf": "page_aware",
            "readme.md": "markdown_heading",
            "tool.py": "source_code",
        }
        for relative_path, expected in expectations.items():
            source = SourceDescriptor(
                source_id=f"source-{relative_path}",
                uri=f"file:///{relative_path}",
                relative_path=relative_path,
                content_hash="fixture",
            )
            with self.subTest(relative_path=relative_path):
                self.assertEqual(service._profile_for(source), expected)

    def test_sidecar_adds_profile_and_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "manual.txt"
            document.write_text("manual", encoding="utf-8")
            (root / "manual.txt.rag.yaml").write_text(
                "profile: markdown_heading\nmetadata:\n  department: support\n",
                encoding="utf-8",
            )
            source = next(iter(FileSystemSourceProvider([document]).discover()))
            configured = apply_sidecar(source)

        self.assertEqual(configured.metadata["rag_profile"], "markdown_heading")
        self.assertEqual(configured.metadata["department"], "support")

    def test_sidecar_profile_overrides_routing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            document = root / "manual.txt"
            document.write_text("manual", encoding="utf-8")
            (root / "manual.txt.rag.yaml").write_text(
                "profile: markdown_heading\n", encoding="utf-8"
            )
            config = RagConfig(
                chunk_profiles=profiles(),
                routing=(RoutingRule(match=RouteMatch(extensions=(".txt",)), use="default"),),
            )
            report = IngestionService(config).preview([document])

        self.assertEqual(report.items[0].chunks[0].metadata["chunker"], "heading_recursive")

    def test_path_and_metadata_route_conditions_both_apply(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            department = root / "support"
            department.mkdir()
            document = department / "faq.txt"
            document.write_text("frequently asked", encoding="utf-8")
            config = RagConfig(
                chunk_profiles=profiles(),
                routing=(
                    RoutingRule(
                        match=RouteMatch(
                            path_patterns=("support/**",), metadata={"department": "support"}
                        ),
                        use="markdown_heading",
                    ),
                ),
            )
            source = next(
                iter(
                    FileSystemSourceProvider(
                        [document], root=root, metadata={"department": "support"}
                    ).discover()
                )
            )
            service = IngestionService(config)
            self.assertEqual(service._profile_for(source), "markdown_heading")


if __name__ == "__main__":
    unittest.main()
