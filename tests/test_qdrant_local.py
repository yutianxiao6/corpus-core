from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from corpuscore.contracts.chunks import Chunk
from corpuscore.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
    SparseEmbeddingSpecification,
    VectorRecord,
)
from corpuscore.contracts.retrieval import SearchRequest, SearchVector
from corpuscore.exceptions import IndexCompatibilityError
from corpuscore.vectorstores import QdrantLocalVectorStore, chunk_to_payload


def make_index_specification() -> IndexSpecification:
    return IndexSpecification(
        index_format_version=1,
        payload_schema_version=1,
        embedding=EmbeddingSpecification(
            provider="test",
            model="fake",
            revision="1",
            dimension=3,
            normalized=True,
            distance=DistanceMetric.COSINE,
            max_length=100,
        ),
        parser_versions={"text": "1"},
        chunking_configuration={"size": 100},
    )


def make_chunk(chunk_id: str, document_id: str, content: str, index: int) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=document_id,
        content=content,
        embedding_text=content,
        source_uri=f"file:///{document_id}.txt",
        chunk_index=index,
        heading_path=("Section",),
        token_count=len(content),
        metadata={"department": "engineering"},
    )


class QdrantLocalTests(unittest.TestCase):
    def test_named_dense_and_sparse_vectors_share_one_collection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specification = replace(
                make_index_specification(),
                sparse_embedding=SparseEmbeddingSpecification(
                    provider="fixture",
                    algorithm="exact",
                    revision="1",
                    hash_space=1000,
                    normalized=True,
                ),
            )
            exact = make_chunk("exact", "doc-exact", "退款政策", 0)
            semantic = make_chunk("semantic", "doc-semantic", "returns", 0)
            with QdrantLocalVectorStore(directory) as store:
                store.ensure_index(specification)
                store.upsert(
                    [
                        VectorRecord(
                            exact.chunk_id,
                            [0.0, 1.0, 0.0],
                            chunk_to_payload(exact),
                            sparse_indices=[7],
                            sparse_values=[1.0],
                        ),
                        VectorRecord(
                            semantic.chunk_id,
                            [1.0, 0.0, 0.0],
                            chunk_to_payload(semantic),
                            sparse_indices=[9],
                            sparse_values=[1.0],
                        ),
                    ]
                )
                dense = store.search(SearchRequest([1.0, 0.0, 0.0], limit=1))
                sparse = store.search(
                    SearchRequest(
                        (),
                        limit=1,
                        sparse_indices=[7],
                        sparse_values=[1.0],
                        vector=SearchVector.SPARSE,
                    )
                )

            self.assertEqual(dense[0].chunk_id, "semantic")
            self.assertEqual(sparse[0].chunk_id, "exact")

    def test_create_upsert_filter_search_delete_and_reopen(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "qdrant"
            specification = make_index_specification()
            first = make_chunk("chunk-a", "doc-a", "closest", 0)
            second = make_chunk("chunk-b", "doc-b", "other", 0)
            with QdrantLocalVectorStore(path) as store:
                store.ensure_index(specification)
                report = store.upsert(
                    [
                        VectorRecord(first.chunk_id, [1.0, 0.0, 0.0], chunk_to_payload(first)),
                        VectorRecord(second.chunk_id, [0.0, 1.0, 0.0], chunk_to_payload(second)),
                    ]
                )
                self.assertEqual(report.completed_count, 2)
                hits = store.search(SearchRequest([1.0, 0.0, 0.0], limit=2))
                self.assertEqual([hit.chunk_id for hit in hits], ["chunk-a", "chunk-b"])
                filtered = store.search(
                    SearchRequest([1.0, 0.0, 0.0], limit=2, filters={"document_id": "doc-b"})
                )
                self.assertEqual([hit.chunk_id for hit in filtered], ["chunk-b"])
                fetched = store.fetch(["chunk-b", "missing", "chunk-a"])
                self.assertEqual([hit.chunk_id for hit in fetched], ["chunk-b", "chunk-a"])
                deleted = store.delete(["chunk-a", "does-not-exist"])
                self.assertEqual(deleted.requested_count, 2)
                self.assertEqual(deleted.deleted_count, 1)

            with QdrantLocalVectorStore(path) as reopened:
                reopened.ensure_index(specification)
                hits = reopened.search(SearchRequest([1.0, 0.0, 0.0], limit=2))
                self.assertEqual([hit.chunk_id for hit in hits], ["chunk-b"])

    def test_incompatible_existing_collection_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specification = make_index_specification()
            with QdrantLocalVectorStore(directory) as store:
                store.ensure_index(specification)
                incompatible = replace(
                    specification,
                    embedding=replace(specification.embedding, query_instruction="new"),
                )
                with self.assertRaises(IndexCompatibilityError):
                    store.ensure_index(incompatible)

    def test_empty_upsert_and_delete_are_noops(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            QdrantLocalVectorStore(directory) as store,
        ):
            self.assertEqual(store.upsert([]).completed_count, 0)
            self.assertEqual(store.delete([]).deleted_count, 0)

    def test_staging_alias_switch_and_rollback_keep_complete_collections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specification = make_index_specification()
            first_chunk = make_chunk("first", "doc-1", "first version", 0)
            second_chunk = make_chunk("second", "doc-2", "second version", 0)
            with QdrantLocalVectorStore(directory, collection_name="active") as store:
                first = store.create_staging(specification, version="v1")
                first.upsert(
                    [VectorRecord("first", [1.0, 0.0, 0.0], chunk_to_payload(first_chunk))]
                )
                store.activate_staging(first)
                self.assertEqual(store.active_collection(), "active__v1")
                self.assertEqual(store.point_count(), 1)

                second = store.create_staging(specification, version="v2")
                second.upsert(
                    [VectorRecord("second", [0.0, 1.0, 0.0], chunk_to_payload(second_chunk))]
                )
                store.activate_staging(second)
                self.assertEqual(store.active_collection(), "active__v2")
                self.assertEqual(store.point_count(), 1)

                store.activate_staging("active__v1")
                self.assertEqual(store.active_collection(), "active__v1")
                hits = store.search(SearchRequest([1.0, 0.0, 0.0], limit=2))
                self.assertEqual([hit.chunk_id for hit in hits], ["first"])


if __name__ == "__main__":
    unittest.main()
