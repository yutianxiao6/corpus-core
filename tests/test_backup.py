from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from qdrant_client import AsyncQdrantClient, QdrantClient

from corpuscore.backup import read_server_backup, restore_local_backup
from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
    VectorRecord,
)
from corpuscore.contracts.retrieval import SearchRequest
from corpuscore.vectorstores import (
    QdrantLocalVectorStore,
    QdrantServerVectorStore,
    chunk_to_payload,
)


def specification() -> IndexSpecification:
    return IndexSpecification(
        index_format_version=1,
        payload_schema_version=1,
        embedding=EmbeddingSpecification(
            provider="fixture",
            model="fixture",
            revision="1",
            dimension=2,
            normalized=True,
            distance=DistanceMetric.COSINE,
            max_length=100,
        ),
        parser_versions={"text": "1"},
        chunking_configuration={"size": 100},
    )


class BackupTests(unittest.TestCase):
    def test_local_backup_restore_preserves_queryable_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            archive = root / "backups" / "index.tar.gz"
            restored = root / "restored"
            chunk = Chunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="backup content",
                embedding_text="backup content",
                source_uri="file:///backup.txt",
                chunk_index=0,
            )
            with QdrantLocalVectorStore(source, collection_name="active") as store:
                store.ensure_index(specification())
                store.upsert([VectorRecord("chunk-1", [1.0, 0.0], chunk_to_payload(chunk))])
                manifest = store.backup(archive, index_fingerprint=specification().fingerprint())
                live_hits = store.search(SearchRequest([1.0, 0.0], limit=1))

            self.assertEqual(manifest["mode"], "local")
            self.assertEqual(live_hits[0].chunk_id, "chunk-1")
            restored_manifest = restore_local_backup(
                archive,
                restored,
                collection_name="active",
                expected_index_fingerprint=specification().fingerprint(),
            )
            self.assertEqual(restored_manifest["source_sha256"], manifest["source_sha256"])
            with QdrantLocalVectorStore(restored, collection_name="active") as store:
                store.ensure_index(specification())
                hits = store.search(SearchRequest([1.0, 0.0], limit=1))
            self.assertEqual(hits[0].chunk_id, "chunk-1")

    def test_server_backup_is_a_validated_snapshot_descriptor(self) -> None:
        client = Mock(spec=QdrantClient)
        client.create_snapshot.return_value = SimpleNamespace(name="snapshot-1")
        async_client = Mock(spec=AsyncQdrantClient)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.json"
            store = QdrantServerVectorStore(
                "http://qdrant.internal:6333",
                collection_name="active",
                client=client,
                async_client=async_client,
            )
            manifest = store.backup(path, index_fingerprint="fingerprint")
            loaded = read_server_backup(
                path,
                collection_name="active",
                expected_index_fingerprint="fingerprint",
            )
        self.assertEqual(manifest, loaded)
        client.create_snapshot.assert_called_once_with("active", wait=True)


if __name__ == "__main__":
    unittest.main()
