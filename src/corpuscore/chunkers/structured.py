"""Page-aware, paragraph-packing and table-row chunk strategies."""

from __future__ import annotations

from collections.abc import Sequence

from corpuscore.chunkers.recursive import RecursiveChunker
from corpuscore.contracts.chunks import ChunkDraft
from corpuscore.contracts.documents import ContentBlock, ContentBlockType, ParsedDocument


class PageAwareChunker(RecursiveChunker):
    name = "page_aware"

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        groups: list[list[ContentBlock]] = []
        current: list[ContentBlock] = []
        current_page: int | None = None
        for block in document.blocks:
            if block.block_type is ContentBlockType.PAGE_BREAK:
                if current:
                    groups.append(current)
                    current = []
                current_page = None
                continue
            if current and block.page != current_page and block.page is not None:
                groups.append(current)
                current = []
            current.append(block)
            current_page = block.page
        if current:
            groups.append(current)
        return tuple(
            draft
            for group in groups
            for draft in self._split_blocks(document, group, chunker_name=self.name)
        )


class ParagraphPackingChunker:
    name = "paragraph_packing"
    version = "1"

    def __init__(self, *, target_size: int = 800, maximum_size: int = 1000) -> None:
        if target_size <= 0 or maximum_size <= 0 or target_size > maximum_size:
            raise ValueError("sizes must be positive and target_size must not exceed maximum_size")
        self.target_size = target_size
        self.maximum_size = maximum_size
        self._splitter = RecursiveChunker(chunk_size=maximum_size, chunk_overlap=0)

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        groups: list[list[ContentBlock]] = []
        current: list[ContentBlock] = []
        current_size = 0
        for block in document.blocks:
            if block.block_type is ContentBlockType.PAGE_BREAK or not block.content.strip():
                continue
            separator_size = 2 if current else 0
            proposed = current_size + separator_size + len(block.content)
            if current and proposed > self.target_size:
                groups.append(current)
                current = []
                current_size = 0
            current.append(block)
            current_size += (2 if current_size else 0) + len(block.content)
        if current:
            groups.append(current)
        return tuple(
            draft
            for group in groups
            for draft in self._splitter._split_blocks(document, group, chunker_name=self.name)
        )


class TableRowChunker:
    name = "table_rows"
    version = "1"

    def __init__(self, *, max_rows_per_chunk: int = 10, repeat_headers: bool = True) -> None:
        if max_rows_per_chunk <= 0:
            raise ValueError("max_rows_per_chunk must be positive")
        self.max_rows_per_chunk = max_rows_per_chunk
        self.repeat_headers = repeat_headers
        self._fallback = RecursiveChunker(chunk_size=1000, chunk_overlap=100)

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        drafts: list[ChunkDraft] = []
        for block in document.blocks:
            if block.block_type is not ContentBlockType.TABLE:
                drafts.extend(
                    self._fallback._split_blocks(document, (block,), chunker_name=self.name)
                )
                continue
            rows = [row for row in block.content.splitlines() if row.strip()]
            if not rows:
                continue
            header, data_rows = rows[0], rows[1:]
            if not data_rows:
                data_rows = [header]
            for offset in range(0, len(data_rows), self.max_rows_per_chunk):
                selected = data_rows[offset : offset + self.max_rows_per_chunk]
                content_rows = (
                    [header] if self.repeat_headers and selected != [header] else []
                ) + selected
                content = "\n".join(content_rows)
                drafts.append(
                    ChunkDraft(
                        document_id=document.document_id,
                        content=content,
                        source_uri=document.source.uri,
                        title=document.title,
                        page_start=block.page,
                        page_end=block.page,
                        char_start=block.char_start,
                        char_end=block.char_end,
                        metadata={
                            "chunker": self.name,
                            "chunker_version": self.version,
                            "table_header": header,
                            "row_start": offset,
                            "row_count": len(selected),
                        },
                    )
                )
        return tuple(drafts)
