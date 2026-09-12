"""Experimental adjacent-block semantic chunking using a local embedder."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from corpuscore.contracts.chunks import ChunkDraft
from corpuscore.contracts.documents import ContentBlock, ContentBlockType, ParsedDocument
from corpuscore.exceptions import ChunkingError
from corpuscore.ports import Vector


@dataclass(frozen=True, slots=True)
class _Unit:
    block: ContentBlock
    content: str
    heading_path: tuple[str, ...]


class SemanticChunker:
    """Group adjacent blocks while their embedding similarity remains high."""

    name = "semantic"
    version = "1-experimental"

    def __init__(
        self,
        embed_documents: Callable[[Sequence[str]], Sequence[Vector]],
        *,
        similarity_threshold: float = 0.45,
        minimum_chunk_size: int = 200,
        maximum_chunk_size: int = 1200,
    ) -> None:
        if not -1 <= similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between -1 and 1")
        if minimum_chunk_size < 0 or maximum_chunk_size <= 0:
            raise ValueError("chunk sizes must be non-negative and maximum must be positive")
        if minimum_chunk_size > maximum_chunk_size:
            raise ValueError("minimum_chunk_size must not exceed maximum_chunk_size")
        self.embed_documents = embed_documents
        self.similarity_threshold = similarity_threshold
        self.minimum_chunk_size = minimum_chunk_size
        self.maximum_chunk_size = maximum_chunk_size

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        units = self._units(document.blocks)
        if not units:
            return ()
        try:
            vectors = tuple(self.embed_documents(tuple(unit.content for unit in units)))
        except Exception as exc:
            raise ChunkingError("semantic chunk embedding failed") from exc
        if len(vectors) != len(units):
            raise ChunkingError("semantic chunk embedder returned an unexpected batch size")
        similarities = tuple(
            self._cosine(vectors[index - 1], vectors[index]) for index in range(1, len(vectors))
        )
        groups: list[list[int]] = []
        current = [0]
        current_size = len(units[0].content)
        for index in range(1, len(units)):
            separator = 2
            proposed = current_size + separator + len(units[index].content)
            semantic_break = (
                similarities[index - 1] < self.similarity_threshold
                and current_size >= self.minimum_chunk_size
            )
            if current and (semantic_break or proposed > self.maximum_chunk_size):
                groups.append(current)
                current = []
                current_size = 0
            current.append(index)
            current_size += (separator if current_size else 0) + len(units[index].content)
        if current:
            groups.append(current)
        self._merge_small_tail(groups, units)
        return tuple(self._draft(document, units, indexes, similarities) for indexes in groups)

    def _draft(
        self,
        document: ParsedDocument,
        units: Sequence[_Unit],
        indexes: Sequence[int],
        similarities: Sequence[float],
    ) -> ChunkDraft:
        selected = [units[index] for index in indexes]
        pages = [unit.block.page for unit in selected if unit.block.page is not None]
        starts = [unit.block.char_start for unit in selected if unit.block.char_start is not None]
        ends = [unit.block.char_end for unit in selected if unit.block.char_end is not None]
        internal = [similarities[index - 1] for index in indexes[1:]]
        return ChunkDraft(
            document_id=document.document_id,
            content="\n\n".join(unit.content for unit in selected),
            source_uri=document.source.uri,
            title=document.title,
            heading_path=selected[0].heading_path,
            page_start=min(pages) if pages else None,
            page_end=max(pages) if pages else None,
            char_start=min(starts) if starts else None,
            char_end=max(ends) if ends else None,
            metadata={
                "chunker": self.name,
                "chunker_version": self.version,
                "block_types": tuple(
                    dict.fromkeys(unit.block.block_type.value for unit in selected)
                ),
                "semantic_unit_count": len(selected),
                "minimum_adjacent_similarity": min(internal) if internal else None,
            },
        )

    def _merge_small_tail(self, groups: list[list[int]], units: Sequence[_Unit]) -> None:
        if len(groups) < 2:
            return
        tail_size = self._group_size(groups[-1], units)
        combined_size = self._group_size(groups[-2] + groups[-1], units)
        if tail_size < self.minimum_chunk_size and combined_size <= self.maximum_chunk_size:
            groups[-2].extend(groups.pop())

    @staticmethod
    def _group_size(indexes: Sequence[int], units: Sequence[_Unit]) -> int:
        return sum(len(units[index].content) for index in indexes) + max(0, len(indexes) - 1) * 2

    def _units(self, blocks: Sequence[ContentBlock]) -> tuple[_Unit, ...]:
        units: list[_Unit] = []
        heading_stack: list[str] = []
        for block in blocks:
            if block.block_type is ContentBlockType.PAGE_BREAK or not block.content.strip():
                continue
            if block.block_type is ContentBlockType.HEADING:
                level = block.heading_level or 1
                heading_stack = heading_stack[: max(0, level - 1)]
                heading_stack.append(block.content)
            content = block.content.strip()
            if len(content) <= self.maximum_chunk_size:
                units.append(_Unit(block, content, tuple(heading_stack)))
                continue
            offset = 0
            while offset < len(content):
                end = min(len(content), offset + self.maximum_chunk_size)
                if end < len(content):
                    boundary = max(
                        content.rfind("\n", offset, end),
                        content.rfind("。", offset, end),
                        content.rfind(". ", offset, end),
                    )
                    if boundary > offset:
                        end = boundary + 1
                part = content[offset:end].strip()
                if part:
                    units.append(_Unit(block, part, tuple(heading_stack)))
                offset = end
        return tuple(units)

    @staticmethod
    def _cosine(left: Vector, right: Vector) -> float:
        if len(left) != len(right) or not left:
            raise ChunkingError("semantic chunk vectors have incompatible dimensions")
        if not all(math.isfinite(float(value)) for value in (*left, *right)):
            raise ChunkingError("semantic chunk vectors contain non-finite values")
        dot = sum(float(a) * float(b) for a, b in zip(left, right, strict=True))
        left_norm = math.sqrt(sum(float(value) ** 2 for value in left))
        right_norm = math.sqrt(sum(float(value) ** 2 for value in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        return dot / (left_norm * right_norm)
