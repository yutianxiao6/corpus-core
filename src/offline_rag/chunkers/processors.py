"""Composable chunk quality and embedding-context processors."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Sequence
from dataclasses import replace

from offline_rag.chunkers.recursive import RecursiveChunker
from offline_rag.contracts.chunks import ChunkDraft


class ChunkSizeProcessor:
    """Drop empty drafts, force maximum size, merge compatible short drafts."""

    def __init__(
        self,
        *,
        minimum_size: int = 80,
        maximum_size: int = 1000,
        length_function: Callable[[str], int] = len,
        count_tokens: bool = False,
    ) -> None:
        if minimum_size < 0:
            raise ValueError("minimum_size must be non-negative")
        if maximum_size <= 0 or minimum_size > maximum_size:
            raise ValueError("maximum_size must be positive and at least minimum_size")
        self.minimum_size = minimum_size
        self.maximum_size = maximum_size
        self.length_function = length_function
        self.count_tokens = count_tokens
        self._fallback = RecursiveChunker(
            chunk_size=maximum_size,
            chunk_overlap=0,
            length_function=length_function,
            count_tokens=count_tokens,
        )

    def process(self, chunks: Sequence[ChunkDraft]) -> Sequence[ChunkDraft]:
        sized: list[ChunkDraft] = []
        for chunk in chunks:
            if not chunk.content.strip():
                continue
            sized.extend(self._force_maximum(chunk))

        merged: list[ChunkDraft] = []
        for chunk in sized:
            if merged and self._should_merge(merged[-1], chunk):
                merged[-1] = self._merge(merged[-1], chunk)
            else:
                merged.append(self._with_count(chunk))
        return tuple(merged)

    def _force_maximum(self, chunk: ChunkDraft) -> tuple[ChunkDraft, ...]:
        if self.length_function(chunk.content) <= self.maximum_size:
            return (self._with_count(chunk),)
        result: list[ChunkDraft] = []
        for content, offset in self._fallback.split_text(chunk.content):
            metadata = dict(chunk.metadata)
            metadata["forced_split"] = True
            start = chunk.char_start + offset if chunk.char_start is not None else None
            end = start + len(content) if start is not None else None
            result.append(
                replace(
                    chunk,
                    content=content,
                    embedding_text=None,
                    char_start=start,
                    char_end=end,
                    token_count=self.length_function(content) if self.count_tokens else None,
                    metadata=metadata,
                )
            )
        return tuple(result)

    def _should_merge(self, previous: ChunkDraft, current: ChunkDraft) -> bool:
        if previous.document_id != current.document_id:
            return False
        if previous.heading_path != current.heading_path:
            return False
        if not (
            self.length_function(previous.content) < self.minimum_size
            or self.length_function(current.content) < self.minimum_size
        ):
            return False
        combined = f"{previous.content}\n\n{current.content}"
        return self.length_function(combined) <= self.maximum_size

    def _merge(self, previous: ChunkDraft, current: ChunkDraft) -> ChunkDraft:
        metadata = dict(previous.metadata)
        prior_merge_count = metadata.get("merged_short_chunks", 0)
        merge_count = prior_merge_count if isinstance(prior_merge_count, int) else 0
        metadata["merged_short_chunks"] = merge_count + 1
        content = f"{previous.content}\n\n{current.content}"
        return replace(
            previous,
            content=content,
            embedding_text=None,
            page_start=self._minimum(previous.page_start, current.page_start),
            page_end=self._maximum(previous.page_end, current.page_end),
            char_start=self._minimum(previous.char_start, current.char_start),
            char_end=self._maximum(previous.char_end, current.char_end),
            token_count=self.length_function(content) if self.count_tokens else None,
            metadata=metadata,
        )

    def _with_count(self, chunk: ChunkDraft) -> ChunkDraft:
        count = self.length_function(chunk.content) if self.count_tokens else chunk.token_count
        return replace(chunk, token_count=count)

    @staticmethod
    def _minimum(first: int | None, second: int | None) -> int | None:
        values = [value for value in (first, second) if value is not None]
        return min(values) if values else None

    @staticmethod
    def _maximum(first: int | None, second: int | None) -> int | None:
        values = [value for value in (first, second) if value is not None]
        return max(values) if values else None


class HeadingContextInjector:
    """Add retrieval context to embedding text without altering source content."""

    def __init__(self, *, separator: str = "\n\n", include_title: bool = True) -> None:
        if not separator:
            raise ValueError("separator must not be empty")
        self._separator = separator
        self._include_title = include_title

    def process(self, chunks: Sequence[ChunkDraft]) -> Sequence[ChunkDraft]:
        result: list[ChunkDraft] = []
        for chunk in chunks:
            context: list[str] = []
            if self._include_title and chunk.title:
                context.append(chunk.title)
            for heading in chunk.heading_path:
                if heading and heading not in context:
                    context.append(heading)
            body = chunk.embedding_text or chunk.content
            embedding_text = self._separator.join((*context, body)) if context else body
            result.append(replace(chunk, embedding_text=embedding_text))
        return tuple(result)


class ChunkDeduplicator:
    """Order-preserving exact and optional near-duplicate removal."""

    def __init__(
        self,
        *,
        approximate_threshold: float | None = None,
        shingle_size: int = 4,
    ) -> None:
        if approximate_threshold is not None and not 0 < approximate_threshold <= 1:
            raise ValueError("approximate_threshold must be in (0, 1]")
        if shingle_size <= 0:
            raise ValueError("shingle_size must be positive")
        self._threshold = approximate_threshold
        self._shingle_size = shingle_size

    def process(self, chunks: Sequence[ChunkDraft]) -> Sequence[ChunkDraft]:
        accepted: list[ChunkDraft] = []
        normalized_values: set[str] = set()
        shingles: list[frozenset[str]] = []
        for chunk in chunks:
            normalized = self._normalize(chunk.content)
            if normalized in normalized_values:
                continue
            candidate_shingles = self._shingles(normalized)
            if self._threshold is not None and any(
                self._jaccard(candidate_shingles, existing) >= self._threshold
                for existing in shingles
            ):
                continue
            accepted.append(chunk)
            normalized_values.add(normalized)
            shingles.append(candidate_shingles)
        return tuple(accepted)

    @staticmethod
    def _normalize(content: str) -> str:
        value = unicodedata.normalize("NFKC", content).casefold()
        return re.sub(r"\s+", " ", value).strip()

    def _shingles(self, content: str) -> frozenset[str]:
        compact = re.sub(r"\s+", "", content)
        if len(compact) <= self._shingle_size:
            return frozenset((compact,))
        return frozenset(
            compact[index : index + self._shingle_size]
            for index in range(len(compact) - self._shingle_size + 1)
        )

    @staticmethod
    def _jaccard(first: frozenset[str], second: frozenset[str]) -> float:
        union = first | second
        return len(first & second) / len(union) if union else 1.0
