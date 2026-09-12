"""CorpusCore public Python API."""

from __future__ import annotations

from typing import TYPE_CHECKING

from corpuscore.config import CorpusConfig, load_config
from corpuscore.contracts.chunks import Chunk, ChunkDraft
from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    ParsedDocument,
    SourceDescriptor,
)
from corpuscore.contracts.indexing import EmbeddingSpecification, IndexSpecification
from corpuscore.contracts.retrieval import (
    Citation,
    QueryOverrides,
    RetrievalCandidate,
    RetrievalResult,
)
from corpuscore.exceptions import CorpusCoreError
from corpuscore.registry import ComponentRegistry

if TYPE_CHECKING:
    from corpuscore.engine import RetrievalEngine

__all__ = [
    "Chunk",
    "ChunkDraft",
    "Citation",
    "ComponentRegistry",
    "ContentBlock",
    "ContentBlockType",
    "CorpusConfig",
    "CorpusCoreError",
    "EmbeddingSpecification",
    "IndexSpecification",
    "ParsedDocument",
    "QueryOverrides",
    "RetrievalCandidate",
    "RetrievalEngine",
    "RetrievalResult",
    "SourceDescriptor",
    "load_config",
]

__version__ = "0.1.0.dev0"


def __getattr__(name: str) -> object:
    """Load the engine lazily so the lightweight contracts remain importable."""

    if name == "RetrievalEngine":
        from corpuscore.engine import RetrievalEngine

        return RetrievalEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
