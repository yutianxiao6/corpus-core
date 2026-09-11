"""Built-in structure-aware chunking strategies and processors."""

from offline_rag.chunkers.finalize import ChunkFinalizer
from offline_rag.chunkers.heading import HeadingRecursiveChunker
from offline_rag.chunkers.processors import ChunkSizeProcessor, HeadingContextInjector
from offline_rag.chunkers.recursive import DEFAULT_SEPARATORS, RecursiveChunker

__all__ = [
    "DEFAULT_SEPARATORS",
    "ChunkFinalizer",
    "ChunkSizeProcessor",
    "HeadingContextInjector",
    "HeadingRecursiveChunker",
    "RecursiveChunker",
]
