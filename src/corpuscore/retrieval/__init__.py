"""Retrieval strategies."""

from corpuscore.retrieval.dense import DenseSimilarityStrategy
from corpuscore.retrieval.fusion import normalized_weighted_fusion, reciprocal_rank_fusion
from corpuscore.retrieval.hybrid import HybridRetrievalStrategy
from corpuscore.retrieval.postprocessing import (
    MaximalMarginalRelevanceSelector,
    NeighborExpander,
    limit_per_document,
    score_threshold_filter,
)
from corpuscore.retrieval.sparse import SparseSimilarityStrategy

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
