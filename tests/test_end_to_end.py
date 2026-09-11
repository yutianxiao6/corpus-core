from __future__ import annotations

import tempfile
import unittest
from collections.abc import Sequence
from pathlib import Path

from offline_rag.config.models import (
    EmbeddingConfig,
    HeadingRecursiveChunkProfile,
    IngestionConfig,
    RagConfig,
    RecursiveChunkProfile,
    VectorStoreConfig,
)
from offline_rag.contracts.indexing import DistanceMetric, EmbeddingSpecification
from offline_rag.contracts.retrieval import RetrievalOptions, RetrievalRequest
from offline_rag.indexing import IngestionJournal
from offline_rag.ingestion import IngestionService, build_index_specification
from offline_rag.ports import Vector
from offline_rag.retrieval import DenseSimilarityStrategy
from offline_rag.vectorstores import QdrantLocalVectorStore


class KeywordEmbedding:
    def __init__(self) -> None:
        self.specification = EmbeddingSpecification(
            provider="fixture",
            model="keyword",
            revision="1",
            dimension=3,
            normalized=True,
            distance=DistanceMetric.COSINE,
            max_length=100,
            model_checksum="fixture",
        )

    def _vector(self, text: str) -> Vector:
        lowered = text.lower()
        if "安装" in text or "install" in lowered:
            return (1.0, 0.0, 0.0)
        if "upgrade" in lowered or "升级" in text:
            return (0.0, 1.0, 0.0)
        return (0.0, 0.0, 1.0)

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
        return tuple(self._vector(text) for text in texts)

    def embed_query(self, text: str) -> Vector:
        return self._vector(text)

    async def aembed_query(self, text: str) -> Vector:
        return self.embed_query(text)


class OfflineEndToEndTests(unittest.TestCase):
    def test_txt_and_markdown_flow_from_directory_to_dense_result(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            (documents / "install.md").write_text(
                "# 安装\n\nInstall the package with uv.", encoding="utf-8"
            )
            (documents / "upgrade.txt").write_text(
                "升级说明\n\nUpgrade after making a backup.", encoding="utf-8"
            )
            config = RagConfig(
                embedding=EmbeddingConfig(dimension=3),
                vector_store=VectorStoreConfig(path=str(root / "qdrant")),
                ingestion=IngestionConfig(commit_batch_size=1),
                chunk_profiles={
                    "default": RecursiveChunkProfile(
                        type="recursive", chunk_size=100, chunk_overlap=10, minimum_size=0
                    ),
                    "markdown_heading": HeadingRecursiveChunkProfile(
                        type="heading_recursive", max_chunk_size=100, chunk_overlap=10
                    ),
                },
            )
            embedding = KeywordEmbedding()
            specification = build_index_specification(config, embedding.specification)
            with QdrantLocalVectorStore(root / "qdrant") as store:
                service = IngestionService(config, embedding=embedding, vector_store=store)
                report = service.build([documents], specification)
                candidates = DenseSimilarityStrategy(embedding, store, fetch_k=5).retrieve(
                    RetrievalRequest(
                        "How do I install it?",
                        RetrievalOptions(profile="fast", final_k=1, organizer="flat"),
                    )
                )

        self.assertEqual(report.indexed_count, 2)
        self.assertEqual(report.failed_count, 0)
        self.assertGreaterEqual(report.chunk_count, 2)
        self.assertEqual(len(candidates), 1)
        self.assertIn("install", candidates[0].chunk.content.lower())

    def test_incremental_sync_handles_unchanged_modified_new_and_deleted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            first = documents / "first.txt"
            removed = documents / "removed.txt"
            first.write_text("original content", encoding="utf-8")
            removed.write_text("remove this content", encoding="utf-8")
            config = RagConfig(
                embedding=EmbeddingConfig(dimension=3),
                vector_store=VectorStoreConfig(path=str(root / "qdrant")),
                chunk_profiles={
                    "default": RecursiveChunkProfile(
                        type="recursive", chunk_size=100, chunk_overlap=10, minimum_size=0
                    )
                },
            )
            embedding = KeywordEmbedding()
            specification = build_index_specification(config, embedding.specification)
            with (
                QdrantLocalVectorStore(root / "qdrant") as store,
                IngestionJournal(root / "journal.sqlite3") as journal,
            ):
                service = IngestionService(
                    config, embedding=embedding, vector_store=store, journal=journal
                )
                initial = service.build([documents], specification)
                unchanged = service.sync([documents], specification)
                old_first = next(
                    item for item in journal.list_sources() if item.relative_path == "first.txt"
                )

                first.write_text("modified install content", encoding="utf-8")
                removed.unlink()
                (documents / "new.txt").write_text("brand new content", encoding="utf-8")
                changed = service.sync([documents], specification)
                indexed = journal.list_sources()

                self.assertEqual(initial.indexed_count, 2)
                self.assertTrue(all(item.stage.value == "skipped" for item in unchanged.items))
                self.assertEqual(
                    sorted(item.stage.value for item in changed.items),
                    ["deleted", "indexed", "indexed"],
                )
                self.assertEqual({item.relative_path for item in indexed}, {"first.txt", "new.txt"})
                current_first = next(item for item in indexed if item.relative_path == "first.txt")
                self.assertNotEqual(current_first.chunk_ids, old_first.chunk_ids)

    def test_failed_modified_source_keeps_previous_journal_version(self) -> None:
        class FailingEmbedding(KeywordEmbedding):
            def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]:
                if any("FAIL" in text for text in texts):
                    raise RuntimeError("fixture failure")
                return super().embed_documents(texts)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            documents = root / "documents"
            documents.mkdir()
            path = documents / "source.txt"
            path.write_text("healthy", encoding="utf-8")
            config = RagConfig(
                embedding=EmbeddingConfig(dimension=3),
                vector_store=VectorStoreConfig(path=str(root / "qdrant")),
                chunk_profiles={
                    "default": RecursiveChunkProfile(
                        type="recursive", chunk_size=100, chunk_overlap=10, minimum_size=0
                    )
                },
            )
            embedding = FailingEmbedding()
            specification = build_index_specification(config, embedding.specification)
            with (
                QdrantLocalVectorStore(root / "qdrant") as store,
                IngestionJournal(root / "journal.sqlite3") as journal,
            ):
                service = IngestionService(
                    config, embedding=embedding, vector_store=store, journal=journal
                )
                service.build([documents], specification)
                prior = journal.list_sources()[0]
                path.write_text("FAIL replacement", encoding="utf-8")
                report = service.sync([documents], specification)
                retained = journal.list_sources()[0]

        self.assertEqual(report.failed_count, 1)
        self.assertEqual(retained.content_hash, prior.content_hash)
        self.assertEqual(retained.chunk_ids, prior.chunk_ids)


if __name__ == "__main__":
    unittest.main()
