"""Fail-closed index fingerprint compatibility checks."""

from __future__ import annotations

from corpuscore.contracts.indexing import IndexSpecification
from corpuscore.exceptions import IndexCompatibilityError


def compare_index_specs(
    expected: IndexSpecification, actual: IndexSpecification
) -> tuple[str, ...]:
    mismatches: list[str] = []
    if expected.index_format_version != actual.index_format_version:
        mismatches.append("index_format_version")
    if expected.payload_schema_version != actual.payload_schema_version:
        mismatches.append("payload_schema_version")
    if expected.embedding.fingerprint() != actual.embedding.fingerprint():
        mismatches.append("embedding")
    if expected.parser_versions != actual.parser_versions:
        mismatches.append("parser_versions")
    if expected.chunking_configuration != actual.chunking_configuration:
        mismatches.append("chunking_configuration")
    return tuple(mismatches)


def assert_index_compatible(expected: IndexSpecification, actual: IndexSpecification) -> None:
    mismatches = compare_index_specs(expected, actual)
    if mismatches:
        raise IndexCompatibilityError(
            "configured retrieval pipeline is incompatible with the active index",
            details={
                "mismatches": mismatches,
                "expected_fingerprint": expected.fingerprint(),
                "actual_fingerprint": actual.fingerprint(),
            },
        )
