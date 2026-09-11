"""Deterministic, dependency-free sparse lexical vectors for mixed Chinese and English."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from collections import Counter
from collections.abc import Sequence
from itertools import pairwise

from offline_rag.contracts.indexing import SparseEmbeddingSpecification
from offline_rag.exceptions import EmbeddingError
from offline_rag.ports import SparseVector

_TOKEN_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]|[a-z0-9]+(?:[._:/+-][a-z0-9]+)*")
_CJK_PATTERN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


class HashedLexicalSparseEmbedding:
    """Hash exact lexical features into Qdrant sparse-vector indices.

    The encoder uses no corpus state and no external model, so indexing and querying
    remain reproducible in air-gapped deployments. Single CJK characters and adjacent
    CJK bigrams preserve Chinese exact-term signals; case-folded word/identifier tokens
    preserve English, numbers, product codes, paths, and API names.
    """

    provider_name = "hashed_lexical"

    def __init__(
        self,
        *,
        hash_space: int = 2_147_483_647,
        normalize: bool = True,
        include_cjk_bigrams: bool = True,
        revision: str = "1",
    ) -> None:
        if hash_space <= 0:
            raise ValueError("hash_space must be positive")
        if not revision.strip():
            raise ValueError("revision must not be empty")
        self._hash_space = hash_space
        self._normalize = normalize
        self._include_cjk_bigrams = include_cjk_bigrams
        self._specification = SparseEmbeddingSpecification(
            provider=self.provider_name,
            algorithm="sha256-sublinear-tf",
            revision=revision,
            hash_space=hash_space,
            normalized=normalize,
            include_cjk_bigrams=include_cjk_bigrams,
        )

    @property
    def specification(self) -> SparseEmbeddingSpecification:
        return self._specification

    def embed_documents(self, texts: Sequence[str]) -> Sequence[SparseVector]:
        return tuple(self._encode(text) for text in texts)

    def embed_query(self, text: str) -> SparseVector:
        return self._encode(text)

    def embed_queries(self, texts: Sequence[str]) -> Sequence[SparseVector]:
        return tuple(self._encode(text) for text in texts)

    async def aembed_query(self, text: str) -> SparseVector:
        return await asyncio.to_thread(self.embed_query, text)

    def tokens(self, text: str) -> tuple[str, ...]:
        """Expose deterministic tokenization for diagnostics and tests."""

        base = tuple(match.group(0) for match in _TOKEN_PATTERN.finditer(text.casefold()))
        if not self._include_cjk_bigrams:
            return base
        features = list(base)
        for left, right in pairwise(base):
            if _CJK_PATTERN.fullmatch(left) and _CJK_PATTERN.fullmatch(right):
                features.append(left + right)
        return tuple(features)

    def _encode(self, text: str) -> SparseVector:
        if not text.strip():
            raise EmbeddingError("sparse embedding input must not be empty")
        tokens = self.tokens(text)
        if not tokens:
            raise EmbeddingError("sparse embedding input contains no searchable token")
        counts: Counter[int] = Counter(self._index(token) for token in tokens)
        values = {index: 1.0 + math.log(count) for index, count in counts.items()}
        if self._normalize:
            norm = math.sqrt(sum(value * value for value in values.values()))
            values = {index: value / norm for index, value in values.items()}
        indices = tuple(sorted(values))
        return indices, tuple(values[index] for index in indices)

    def _index(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:8], "big") % self._hash_space
