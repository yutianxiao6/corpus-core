"""Retrieval strategies."""

from offline_rag.retrieval.dense import DenseSimilarityStrategy
from offline_rag.retrieval.fusion import normalized_weighted_fusion, reciprocal_rank_fusion
from offline_rag.retrieval.hybrid import HybridRetrievalStrategy
from offline_rag.retrieval.sparse import SparseSimilarityStrategy

__all__ = [
    "DenseSimilarityStrategy",
    "HybridRetrievalStrategy",
    "SparseSimilarityStrategy",
    "normalized_weighted_fusion",
    "reciprocal_rank_fusion",
]
