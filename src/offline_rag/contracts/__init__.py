"""Stable data contracts used across the retrieval engine."""

from offline_rag.contracts.chunks import Chunk, ChunkDraft
from offline_rag.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
    SourceDescriptor,
)
from offline_rag.contracts.indexing import (
    DeleteReport,
    EmbeddingSpecification,
    IndexSpecification,
    IngestionReport,
    SparseEmbeddingSpecification,
    UpsertReport,
    VectorRecord,
)
from offline_rag.contracts.retrieval import (
    Citation,
    OrganizedResult,
    ResultGroup,
    RetrievalCandidate,
    RetrievalResult,
    SearchHit,
    SearchRequest,
    SearchVector,
)

__all__ = [
    "Chunk",
    "ChunkDraft",
    "Citation",
    "ContentBlock",
    "ContentBlockType",
    "DeleteReport",
    "EmbeddingSpecification",
    "IndexSpecification",
    "IngestionReport",
    "LoadedContent",
    "OrganizedResult",
    "ParsedDocument",
    "ResultGroup",
    "RetrievalCandidate",
    "RetrievalResult",
    "SearchHit",
    "SearchRequest",
    "SearchVector",
    "SourceDescriptor",
    "SparseEmbeddingSpecification",
    "UpsertReport",
    "VectorRecord",
]
