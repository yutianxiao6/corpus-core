"""Built-in structure-aware chunking strategies and processors."""

from corpuscore.chunkers.finalize import ChunkFinalizer
from corpuscore.chunkers.heading import HeadingRecursiveChunker
from corpuscore.chunkers.parent_child import ParentChildChunker
from corpuscore.chunkers.processors import (
    ChunkDeduplicator,
    ChunkSizeProcessor,
    HeadingContextInjector,
)
from corpuscore.chunkers.recursive import DEFAULT_SEPARATORS, RecursiveChunker
from corpuscore.chunkers.semantic import SemanticChunker
from corpuscore.chunkers.structured import (
    PageAwareChunker,
    ParagraphPackingChunker,
    TableRowChunker,
)
from corpuscore.chunkers.syntax import SyntaxChunker

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
    "SemanticChunker",
    "SyntaxChunker",
    "TableRowChunker",
]
