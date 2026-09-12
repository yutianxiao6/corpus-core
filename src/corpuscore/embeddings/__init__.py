"""Local-only embedding providers."""

from corpuscore.embeddings.qwen import QwenSentenceTransformerEmbedding
from corpuscore.embeddings.sparse import HashedLexicalSparseEmbedding

__all__ = ["HashedLexicalSparseEmbedding", "QwenSentenceTransformerEmbedding"]
