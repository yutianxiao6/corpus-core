from __future__ import annotations

import unittest
from unittest.mock import patch

from qdrant_client import QdrantClient

from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
    VectorRecord,
)
from offline_rag.contracts.retrieval import SearchRequest
from offline_rag.vectorstores import QdrantServerVectorStore, chunk_to_payload


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


def record(chunk_id: str) -> VectorRecord:
    chunk = Chunk(
        chunk_id=chunk_id,
        document_id="doc",
        content="server content",
        embedding_text="server content",
        source_uri="file:///server.txt",
        chunk_index=0,
    )
    return VectorRecord(chunk_id, [1.0, 0.0], chunk_to_payload(chunk))


class QdrantServerTests(unittest.TestCase):
    def test_server_adapter_uses_same_staging_alias_and_query_contract(self) -> None:
        client = QdrantClient(":memory:")
        try:
            store = QdrantServerVectorStore(
                "http://qdrant.internal:6333",
                collection_name="active",
                client=client,
            )
            staging = store.create_staging(specification(), version="v1")
            self.assertIsInstance(staging, QdrantServerVectorStore)
            staging.upsert([record("chunk-1")])
            store.activate_staging(staging)

            hits = store.search(SearchRequest([1.0, 0.0], limit=1))

            self.assertEqual(store.active_collection(), "active__v1")
            self.assertEqual(hits[0].chunk_id, "chunk-1")
        finally:
            client.close()

    def test_client_configuration_disables_cloud_inference(self) -> None:
        with patch("offline_rag.vectorstores.qdrant_server.QdrantClient") as factory:
            store = QdrantServerVectorStore(
                "https://qdrant.internal",
                api_key="secret",
                prefer_grpc=True,
                timeout_seconds=9,
                pool_size=20,
            )
            store.close()

        options = factory.call_args.kwargs
        self.assertEqual(options["url"], "https://qdrant.internal")
        self.assertEqual(options["api_key"], "secret")
        self.assertIs(options["prefer_grpc"], True)
        self.assertEqual(options["timeout"], 9)
        self.assertEqual(options["pool_size"], 20)
        self.assertIs(options["cloud_inference"], False)


if __name__ == "__main__":
    unittest.main()
