"""Public exception hierarchy with stable machine-readable error codes."""

from __future__ import annotations

from collections.abc import Mapping

from corpuscore.contracts.common import JSONValue, freeze_metadata


class CorpusCoreError(Exception):
    """Base class for expected errors exposed by the public API."""

    code = "corpuscore_error"

    def __init__(
        self,
        message: str,
        *,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.details: Mapping[str, JSONValue] = freeze_metadata(details)

    def __str__(self) -> str:
        return self.message


class ConfigurationError(CorpusCoreError):
    code = "configuration_error"


class LocalResourceMissingError(CorpusCoreError):
    code = "local_resource_missing"


class UnsupportedDocumentError(CorpusCoreError):
    code = "unsupported_document"


class SourceDiscoveryError(CorpusCoreError):
    code = "source_discovery_error"


class DocumentLoadError(CorpusCoreError):
    code = "document_load_error"


class DocumentParseError(CorpusCoreError):
    code = "document_parse_error"


class ChunkingError(CorpusCoreError):
    code = "chunking_error"


class EmbeddingError(CorpusCoreError):
    code = "embedding_error"


class IndexCompatibilityError(CorpusCoreError):
    code = "index_compatibility_error"


class VectorStoreError(CorpusCoreError):
    code = "vector_store_error"


class RetrievalError(CorpusCoreError):
    code = "retrieval_error"


class RerankerError(CorpusCoreError):
    code = "reranker_error"


class EvaluationRegressionError(CorpusCoreError):
    code = "evaluation_regression"


class ContextBudgetError(CorpusCoreError):
    code = "context_budget_error"


class ConcurrentAccessError(CorpusCoreError):
    code = "concurrent_access"


class RegistryError(CorpusCoreError):
    code = "registry_error"
