"""Offline, configurable retrieval core for RAG systems."""

from offline_rag.contracts.chunks import Chunk, ChunkDraft
from offline_rag.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    ParsedDocument,
    SourceDescriptor,
)
from offline_rag.contracts.indexing import EmbeddingSpecification, IndexSpecification
from offline_rag.contracts.retrieval import (
    Citation,
    QueryOverrides,
    RetrievalCandidate,
    RetrievalResult,
)
from offline_rag.exceptions import OfflineRagError
from offline_rag.registry import ComponentRegistry

__all__ = [
    "Chunk",
    "ChunkDraft",
    "Citation",
    "ComponentRegistry",
    "ContentBlock",
    "ContentBlockType",
    "EmbeddingSpecification",
    "IndexSpecification",
    "OfflineRagError",
    "ParsedDocument",
    "QueryOverrides",
    "RetrievalCandidate",
    "RetrievalResult",
    "SourceDescriptor",
]

__version__ = "0.1.0.dev0"
