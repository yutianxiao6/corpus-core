from __future__ import annotations

import unittest
from dataclasses import replace

from offline_rag.config.models import RagConfig, RouteMatch, RoutingRule
from offline_rag.contracts.indexing import (
    DistanceMetric,
    EmbeddingSpecification,
    IndexSpecification,
)
from offline_rag.exceptions import IndexCompatibilityError
from offline_rag.indexing import assert_index_compatible, compare_index_specs
from offline_rag.ingestion import build_index_specification


def make_specification() -> IndexSpecification:
    embedding = EmbeddingSpecification(
        provider="qwen_sentence_transformers",
        model="Qwen3-Embedding-0.6B",
        revision="pinned-local",
        dimension=1024,
        normalized=True,
        distance=DistanceMetric.COSINE,
        max_length=1024,
        query_instruction="Retrieve relevant passages.",
        tokenizer_revision="pinned-local",
        model_checksum="abc123",
    )
    return IndexSpecification(
        index_format_version=1,
        payload_schema_version=1,
        embedding=embedding,
        parser_versions={"markdown": "1"},
        chunking_configuration={
            "chunker": {"type": "recursive", "size": 800},
            "processors": ["size", "heading_context"],
        },
    )


class IndexCompatibilityTests(unittest.TestCase):
    def test_routing_changes_are_part_of_index_fingerprint(self) -> None:
        embedding = make_specification().embedding
        first = build_index_specification(RagConfig(), embedding)
        second = build_index_specification(
            RagConfig(
                routing=(
                    RoutingRule(
                        match=RouteMatch(extensions=(".txt",)),
                        use="markdown_heading",
                    ),
                )
            ),
            embedding,
        )

        self.assertNotEqual(first.fingerprint(), second.fingerprint())

    def test_identical_specifications_are_compatible(self) -> None:
        specification = make_specification()
        self.assertEqual(compare_index_specs(specification, specification), ())
        assert_index_compatible(specification, specification)

    def test_embedding_or_chunking_change_requires_rebuild(self) -> None:
        expected = make_specification()
        changed_embedding = replace(expected.embedding, query_instruction="Different instruction")
        actual = replace(
            expected,
            embedding=changed_embedding,
            chunking_configuration={"chunker": {"type": "recursive", "size": 500}},
        )
        with self.assertRaises(IndexCompatibilityError) as raised:
            assert_index_compatible(expected, actual)

        self.assertEqual(
            raised.exception.details["mismatches"], ("embedding", "chunking_configuration")
        )
        self.assertNotEqual(
            raised.exception.details["expected_fingerprint"],
            raised.exception.details["actual_fingerprint"],
        )


if __name__ == "__main__":
    unittest.main()
