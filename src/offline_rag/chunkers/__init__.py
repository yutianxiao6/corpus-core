"""Built-in structure-aware chunking strategies and processors."""

from offline_rag.chunkers.finalize import ChunkFinalizer
from offline_rag.chunkers.heading import HeadingRecursiveChunker
from offline_rag.chunkers.parent_child import ParentChildChunker
from offline_rag.chunkers.processors import (
    ChunkDeduplicator,
    ChunkSizeProcessor,
    HeadingContextInjector,
)
from offline_rag.chunkers.recursive import DEFAULT_SEPARATORS, RecursiveChunker
from offline_rag.chunkers.structured import (
    PageAwareChunker,
    ParagraphPackingChunker,
    TableRowChunker,
)

__all__ = [
    "DEFAULT_SEPARATORS",
    "ChunkDeduplicator",
    "ChunkFinalizer",
    "ChunkSizeProcessor",
    "HeadingContextInjector",
    "HeadingRecursiveChunker",
    "PageAwareChunker",
    "ParagraphPackingChunker",
    "ParentChildChunker",
    "RecursiveChunker",
    "TableRowChunker",
]
