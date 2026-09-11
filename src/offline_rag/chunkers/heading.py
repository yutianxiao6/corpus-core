"""Heading-first recursive chunking for Markdown-like structured documents."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from offline_rag.chunkers.recursive import DEFAULT_SEPARATORS, RecursiveChunker
from offline_rag.contracts.chunks import ChunkDraft
from offline_rag.contracts.documents import ContentBlock, ContentBlockType, ParsedDocument


@dataclass(frozen=True, slots=True)
class _Section:
    heading_path: tuple[str, ...]
    blocks: tuple[ContentBlock, ...]


class HeadingRecursiveChunker(RecursiveChunker):
    name = "heading_recursive"

    def __init__(
        self,
        *,
        max_chunk_size: int = 1000,
        chunk_overlap: int = 100,
    ) -> None:
        super().__init__(
            chunk_size=max_chunk_size,
            chunk_overlap=chunk_overlap,
            separators=DEFAULT_SEPARATORS,
        )

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for section in self._sections(document.blocks):
            drafts.extend(
                self._split_blocks(
                    document,
                    section.blocks,
                    chunker_name=self.name,
                    fixed_heading_path=section.heading_path,
                )
            )
        return tuple(drafts)

    @staticmethod
    def _sections(blocks: Sequence[ContentBlock]) -> tuple[_Section, ...]:
        sections: list[_Section] = []
        current: list[ContentBlock] = []
        current_path: tuple[str, ...] = ()
        heading_stack: list[str] = []
        for block in blocks:
            if block.block_type is ContentBlockType.HEADING:
                if current:
                    sections.append(_Section(current_path, tuple(current)))
                    current = []
                level = block.heading_level or 1
                heading_stack = heading_stack[: max(0, level - 1)]
                heading_stack.append(block.content)
                current_path = tuple(heading_stack)
            current.append(block)
        if current:
            sections.append(_Section(current_path, tuple(current)))
        return tuple(sections)
