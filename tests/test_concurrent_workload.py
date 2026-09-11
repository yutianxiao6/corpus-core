from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
    VectorRecord,
)
from offline_rag.contracts.retrieval import SearchRequest
from offline_rag.vectorstores import QdrantLocalVectorStore, chunk_to_payload


def _specification() -> IndexSpecification:
    return IndexSpecification(
        index_format_version=1,
        payload_schema_version=1,
        embedding=EmbeddingSpecification(
            provider="fixture",
            model="fixture",
            revision="1",
            dimension=3,
            normalized=True,
            distance=DistanceMetric.COSINE,
            max_length=100,
        ),
        parser_versions={"text": "1"},
        chunking_configuration={"size": 100},
    )


def _record(index: int) -> VectorRecord:
    chunk = Chunk(
        chunk_id=f"chunk-{index}",
        document_id=f"doc-{index // 4}",
        content=f"parallel content {index}",
        embedding_text=f"parallel content {index}",
        source_uri=f"file:///doc-{index // 4}.txt",
        chunk_index=index % 4,
    )
    return VectorRecord(
        chunk.chunk_id,
        [1.0, float(index % 3), 0.0],
        chunk_to_payload(chunk),
    )


class ConcurrentWorkloadTests(unittest.TestCase):
    def test_queries_and_import_batches_are_safe_in_parallel(self) -> None:
        with (
            tempfile.TemporaryDirectory() as directory,
            QdrantLocalVectorStore(Path(directory) / "qdrant") as store,
        ):
            store.ensure_index(_specification())

            def import_batch(batch: list[VectorRecord]) -> int:
                store.upsert(batch)
                return len(batch)

            def query_repeatedly() -> int:
                count = 0
                for _ in range(12):
                    hits = store.search(SearchRequest([1.0, 0.0, 0.0], limit=5))
                    self.assertLessEqual(len(hits), 5)
                    count += len(hits)
                return count

            records = [_record(index) for index in range(96)]
            batches = [records[offset : offset + 8] for offset in range(0, len(records), 8)]
            with ThreadPoolExecutor(max_workers=16) as executor:
                imports = [executor.submit(import_batch, batch) for batch in batches]
                queries = [executor.submit(query_repeatedly) for _ in range(8)]
                self.assertEqual(sum(future.result() for future in imports), len(records))
                self.assertTrue(all(future.result() >= 0 for future in queries))

            self.assertEqual(store.point_count(), len(records))
            final_hits = store.search(SearchRequest([1.0, 0.0, 0.0], limit=5))
            self.assertEqual(len(final_hits), 5)


if __name__ == "__main__":
    unittest.main()
