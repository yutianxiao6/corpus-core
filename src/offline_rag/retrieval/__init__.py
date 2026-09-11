"""Retrieval strategies."""

from offline_rag.retrieval.dense import DenseSimilarityStrategy
from offline_rag.retrieval.fusion import normalized_weighted_fusion, reciprocal_rank_fusion
from offline_rag.retrieval.hybrid import HybridRetrievalStrategy
from offline_rag.retrieval.postprocessing import (
    MaximalMarginalRelevanceSelector,
    NeighborExpander,
    limit_per_document,
    score_threshold_filter,
)
from offline_rag.retrieval.sparse import SparseSimilarityStrategy

__all__ = [
    "DenseSimilarityStrategy",
    "HybridRetrievalStrategy",
    "MaximalMarginalRelevanceSelector",
    "NeighborExpander",
    "SparseSimilarityStrategy",
    "limit_per_document",
    "normalized_weighted_fusion",
    "reciprocal_rank_fusion",
    "score_threshold_filter",
]
