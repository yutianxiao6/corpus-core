"""Stable data contracts used across the retrieval engine."""

from corpuscore.contracts.chunks import Chunk, ChunkDraft
from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
    SourceDescriptor,
)
from corpuscore.contracts.indexing import (
    DeleteReport,
    EmbeddingSpecification,
    IndexSpecification,
    IngestionReport,
    SparseEmbeddingSpecification,
    UpsertReport,
    VectorRecord,
)
from corpuscore.contracts.retrieval import (
    Citation,
    OrganizedResult,
    QueryOverrides,
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
    "QueryOverrides",
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
