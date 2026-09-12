"""LangChain-backed recursive chunking with source-structure propagation."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from corpuscore.contracts.chunks import ChunkDraft
from corpuscore.contracts.documents import ContentBlock, ContentBlockType, ParsedDocument
from corpuscore.exceptions import ChunkingError

DEFAULT_SEPARATORS = (
    "\n\n",
    "\n",
    "。",
    "！",
    "？",
    "；",
    ".",
    "!",
    "?",
    ";",
    "，",
    ",",
    " ",
    "",
)


@dataclass(frozen=True, slots=True)
class _Segment:
    start: int
    end: int
    block: ContentBlock
    heading_path: tuple[str, ...]


class RecursiveChunker:
    name = "recursive"
    version = "1"

    def __init__(
        self,
        *,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
        separators: Sequence[str] = DEFAULT_SEPARATORS,
        length_function: Callable[[str], int] = len,
        count_tokens: bool = False,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size")
        if not separators:
            raise ValueError("separators must not be empty")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.count_tokens = count_tokens
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators),
            keep_separator="end",
            length_function=length_function,
            add_start_index=True,
            strip_whitespace=True,
        )

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        return self._split_blocks(document, document.blocks, chunker_name=self.name)

    def split_text(self, text: str) -> tuple[tuple[str, int], ...]:
        documents = self._splitter.create_documents([text])
        return tuple(
            (item.page_content, int(item.metadata.get("start_index", 0))) for item in documents
        )

    def _split_blocks(
        self,
        document: ParsedDocument,
        blocks: Sequence[ContentBlock],
        *,
        chunker_name: str,
        fixed_heading_path: tuple[str, ...] | None = None,
    ) -> tuple[ChunkDraft, ...]:
        text, segments = self._compose(blocks)
        if not text.strip():
            return ()
        drafts: list[ChunkDraft] = []
        for content, start in self.split_text(text):
            end = start + len(content)
            overlapping = [
                segment for segment in segments if segment.end > start and segment.start < end
            ]
            if not overlapping:
                raise ChunkingError("splitter produced a chunk outside composed document text")
            pages = [
                segment.block.page for segment in overlapping if segment.block.page is not None
            ]
            source_starts = [
                segment.block.char_start
                for segment in overlapping
                if segment.block.char_start is not None
            ]
            source_ends = [
                segment.block.char_end
                for segment in overlapping
                if segment.block.char_end is not None
            ]
            heading_path = fixed_heading_path or self._heading_at(segments, start)
            block_types = tuple(
                dict.fromkeys(segment.block.block_type.value for segment in overlapping)
            )
            drafts.append(
                ChunkDraft(
                    document_id=document.document_id,
                    content=content,
                    source_uri=document.source.uri,
                    title=document.title,
                    heading_path=heading_path,
                    page_start=min(pages) if pages else None,
                    page_end=max(pages) if pages else None,
                    char_start=min(source_starts) if source_starts else None,
                    char_end=max(source_ends) if source_ends else None,
                    token_count=self.length_function(content) if self.count_tokens else None,
                    metadata={
                        "chunker": chunker_name,
                        "chunker_version": self.version,
                        "block_types": block_types,
                    },
                )
            )
        return tuple(drafts)

    @staticmethod
    def _compose(blocks: Sequence[ContentBlock]) -> tuple[str, tuple[_Segment, ...]]:
        parts: list[str] = []
        segments: list[_Segment] = []
        heading_stack: list[str] = []
        offset = 0
        for block in blocks:
            if block.block_type is ContentBlockType.PAGE_BREAK or not block.content.strip():
                continue
            if parts:
                parts.append("\n\n")
                offset += 2
            if block.block_type is ContentBlockType.HEADING:
                level = block.heading_level or 1
                heading_stack = heading_stack[: max(0, level - 1)]
                heading_stack.append(block.content)
            start = offset
            parts.append(block.content)
            offset += len(block.content)
            segments.append(_Segment(start, offset, block, tuple(heading_stack)))
        return "".join(parts), tuple(segments)

    @staticmethod
    def _heading_at(segments: Sequence[_Segment], position: int) -> tuple[str, ...]:
        heading_path: tuple[str, ...] = ()
        for segment in segments:
            if segment.start > position:
                break
            heading_path = segment.heading_path
        return heading_path
