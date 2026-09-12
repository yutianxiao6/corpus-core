from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, patch

from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client.http import models

from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
    VectorRecord,
)
from corpuscore.contracts.retrieval import SearchRequest
from corpuscore.vectorstores import QdrantServerVectorStore, chunk_to_payload


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
        async_client = AsyncMock(spec=AsyncQdrantClient)
        try:
            store = QdrantServerVectorStore(
                "http://qdrant.internal:6333",
                collection_name="active",
                client=client,
                async_client=async_client,
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
        with (
            patch("corpuscore.vectorstores.qdrant_server.QdrantClient") as factory,
            patch("corpuscore.vectorstores.qdrant_server.AsyncQdrantClient") as async_factory,
        ):
            async_factory.return_value.close = AsyncMock()
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
        self.assertIs(options["check_compatibility"], False)
        self.assertEqual(async_factory.call_args.kwargs, options)


class QdrantServerAsyncTests(unittest.IsolatedAsyncioTestCase):
    async def test_native_async_search_and_fetch_preserve_contract(self) -> None:
        client = QdrantClient(":memory:")
        async_client = AsyncMock(spec=AsyncQdrantClient)
        payload = chunk_to_payload(
            Chunk(
                chunk_id="chunk-1",
                document_id="doc",
                content="async server content",
                embedding_text="async server content",
                source_uri="file:///async.txt",
                chunk_index=0,
            )
        )
        payload["chunk_id"] = "chunk-1"
        async_client.query_points.return_value = models.QueryResponse(
            points=[models.ScoredPoint(id=1, version=0, score=0.9, payload=payload)]
        )
        async_client.retrieve.return_value = [models.Record(id=1, payload=payload)]
        store = QdrantServerVectorStore(
            "http://qdrant.internal:6333",
            client=client,
            async_client=async_client,
        )
        try:
            hits = await store.asearch(SearchRequest([1.0, 0.0], limit=1))
            fetched = await store.afetch(["chunk-1"])
        finally:
            await store.aclose()
            client.close()

        self.assertEqual(hits[0].score, 0.9)
        self.assertEqual(fetched[0].chunk_id, "chunk-1")
        async_client.query_points.assert_awaited_once()
        async_client.retrieve.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
