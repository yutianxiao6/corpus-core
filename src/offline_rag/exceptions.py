"""Public exception hierarchy with stable machine-readable error codes."""

from __future__ import annotations

from collections.abc import Mapping

from offline_rag.contracts.common import JSONValue, freeze_metadata


class OfflineRagError(Exception):
    """Base class for expected errors exposed by the public API."""

    code = "offline_rag_error"

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


class ConfigurationError(OfflineRagError):
    code = "configuration_error"


class OfflineResourceMissingError(OfflineRagError):
    code = "offline_resource_missing"


class UnsupportedDocumentError(OfflineRagError):
    code = "unsupported_document"


class SourceDiscoveryError(OfflineRagError):
    code = "source_discovery_error"


class DocumentLoadError(OfflineRagError):
    code = "document_load_error"


class DocumentParseError(OfflineRagError):
    code = "document_parse_error"


class ChunkingError(OfflineRagError):
    code = "chunking_error"


class EmbeddingError(OfflineRagError):
    code = "embedding_error"


class IndexCompatibilityError(OfflineRagError):
    code = "index_compatibility_error"


class VectorStoreError(OfflineRagError):
    code = "vector_store_error"


class RetrievalError(OfflineRagError):
    code = "retrieval_error"


class RerankerError(OfflineRagError):
    code = "reranker_error"


class ContextBudgetError(OfflineRagError):
    code = "context_budget_error"


class ConcurrentAccessError(OfflineRagError):
    code = "concurrent_access"


class RegistryError(OfflineRagError):
    code = "registry_error"
