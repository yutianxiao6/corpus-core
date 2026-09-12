from __future__ import annotations

import unittest
from dataclasses import FrozenInstanceError

from corpuscore import CorpusConfig, CorpusCoreError, RetrievalEngine, load_config
from corpuscore.contracts.chunks import Chunk, ChunkDraft
from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    ParsedDocument,
    SourceDescriptor,
)
from corpuscore.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
    IngestionItemResult,
    IngestionReport,
    IngestionStage,
    ItemFailure,
)
from corpuscore.contracts.retrieval import (
    ContextBudget,
    QueryOverrides,
    RetrievalOptions,
    RetrievalRequest,
)


def make_source() -> SourceDescriptor:
    return SourceDescriptor(
        source_id="source-1",
        uri="manual.md",
        content_hash="abc123",
        metadata={"labels": ["manual", "zh"]},
    )


class PublicApiTests(unittest.TestCase):
    def test_top_level_library_api_exposes_primary_entry_points(self) -> None:
        self.assertEqual(RetrievalEngine.__name__, "RetrievalEngine")
        self.assertEqual(CorpusConfig.__name__, "CorpusConfig")
        self.assertTrue(issubclass(CorpusCoreError, Exception))
        self.assertTrue(callable(load_config))


class DocumentContractTests(unittest.TestCase):
    def test_nested_metadata_is_immutable(self) -> None:
        source = make_source()

        with self.assertRaises(TypeError):
            source.metadata["new"] = "value"  # type: ignore[index]

        labels = source.metadata["labels"]
        self.assertEqual(labels, ("manual", "zh"))

    def test_parsed_document_freezes_sequences(self) -> None:
        block = ContentBlock(ContentBlockType.PARAGRAPH, "Hello 世界", page=1)
        document = ParsedDocument(
            document_id="doc-1",
            source=make_source(),
            blocks=[block],
            parser_name="markdown",
            parser_version="1",
            language_hints=["zh", "en"],
        )

        self.assertIsInstance(document.blocks, tuple)
        self.assertEqual(document.language_hints, ("zh", "en"))
        with self.assertRaises(FrozenInstanceError):
            document.title = "changed"  # type: ignore[misc]

    def test_invalid_location_range_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "page_end"):
            ChunkDraft(
                document_id="doc-1",
                content="content",
                source_uri="manual.pdf",
                page_start=3,
                page_end=2,
            )

    def test_embedding_text_falls_back_to_content(self) -> None:
        draft = ChunkDraft(document_id="doc-1", content="content", source_uri="manual.md")
        self.assertEqual(draft.text_for_embedding, "content")

    def test_chunk_index_must_be_non_negative(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_index"):
            Chunk(
                chunk_id="chunk-1",
                document_id="doc-1",
                content="content",
                embedding_text="heading\ncontent",
                source_uri="manual.md",
                chunk_index=-1,
            )


class IndexContractTests(unittest.TestCase):
    def make_embedding(self) -> EmbeddingSpecification:
        return EmbeddingSpecification(
            provider="qwen_sentence_transformers",
            model="Qwen3-Embedding-0.6B",
            revision="local-sha256",
            dimension=1024,
            normalized=True,
            distance=DistanceMetric.COSINE,
            max_length=1024,
            query_instruction="Retrieve relevant passages.",
        )

    def test_embedding_fingerprint_is_deterministic(self) -> None:
        first = self.make_embedding()
        second = self.make_embedding()
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_index_fingerprint_ignores_mapping_order(self) -> None:
        first = IndexSpecification(
            index_format_version=1,
            payload_schema_version=1,
            embedding=self.make_embedding(),
            parser_versions={"markdown": "1", "text": "1"},
            chunking_configuration={"size": 800, "separators": ["\n", "。"]},
        )
        second = IndexSpecification(
            index_format_version=1,
            payload_schema_version=1,
            embedding=self.make_embedding(),
            parser_versions={"text": "1", "markdown": "1"},
            chunking_configuration={"separators": ["\n", "。"], "size": 800},
        )
        self.assertEqual(first.fingerprint(), second.fingerprint())

    def test_failed_item_requires_failure_details(self) -> None:
        with self.assertRaisesRegex(ValueError, "failure details"):
            IngestionItemResult(source_uri="bad.pdf", stage=IngestionStage.FAILED)

    def test_ingestion_report_aggregates_results(self) -> None:
        failure = ItemFailure(
            source_uri="bad.pdf",
            stage=IngestionStage.PARSED,
            error_code="parse_failed",
            message="could not parse",
        )
        report = IngestionReport(
            job_id="job-1",
            index_version="v1",
            duration_ms=12.0,
            items=[
                IngestionItemResult(
                    source_uri="good.md",
                    stage=IngestionStage.INDEXED,
                    chunk_count=3,
                ),
                IngestionItemResult(
                    source_uri="bad.pdf",
                    stage=IngestionStage.FAILED,
                    failure=failure,
                ),
            ],
        )
        self.assertEqual(report.indexed_count, 1)
        self.assertEqual(report.failed_count, 1)
        self.assertEqual(report.chunk_count, 3)


class RetrievalContractTests(unittest.TestCase):
    def test_context_budget_requires_limit(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            ContextBudget()

    def test_retrieval_request_rejects_blank_query(self) -> None:
        with self.assertRaisesRegex(ValueError, "query"):
            RetrievalRequest(" ")

    def test_options_freeze_filters(self) -> None:
        options = RetrievalOptions(filters={"source": ["a.md", "b.md"]})
        self.assertEqual(options.filters["source"], ("a.md", "b.md"))

    def test_query_overrides_validate_safe_runtime_values(self) -> None:
        overrides = QueryOverrides(final_k=3, mmr_lambda=0.6, filters={"acl": ["a", "b"]})
        self.assertEqual(overrides.filters["acl"], ("a", "b"))
        with self.assertRaises(ValueError):
            QueryOverrides(mmr_lambda=1.1)


if __name__ == "__main__":
    unittest.main()
