"""Local-only embedding providers."""

from offline_rag.embeddings.qwen import QwenSentenceTransformerEmbedding
from offline_rag.embeddings.sparse import HashedLexicalSparseEmbedding

__all__ = ["HashedLexicalSparseEmbedding", "QwenSentenceTransformerEmbedding"]
