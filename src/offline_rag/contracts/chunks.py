"""Chunk drafts and immutable indexed chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from offline_rag.contracts.common import (
    JSONValue,
    freeze_metadata,
    require_non_empty,
    require_non_negative,
    require_range,
)


@dataclass(frozen=True, slots=True)
class ChunkDraft:
    """A chunk before deterministic identifiers and neighbor links are assigned."""

    document_id: str
    content: str
    source_uri: str
    embedding_text: str | None = None
    title: str | None = None
    heading_path: Sequence[str] = ()
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    parent_key: str | None = None
    token_count: int | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.document_id, "document_id")
        require_non_empty(self.content, "content")
        require_non_empty(self.source_uri, "source_uri")
        if self.embedding_text is not None:
            require_non_empty(self.embedding_text, "embedding_text")
        require_range(self.page_start, self.page_end, "page_start", "page_end")
        require_range(self.char_start, self.char_end, "char_start", "char_end")
        require_non_negative(self.token_count, "token_count")
        object.__setattr__(self, "heading_path", tuple(self.heading_path))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))

    @property
    def text_for_embedding(self) -> str:
        return self.embedding_text or self.content


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    content: str
    embedding_text: str
    source_uri: str
    chunk_index: int
    title: str | None = None
    heading_path: Sequence[str] = ()
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    parent_id: str | None = None
    previous_id: str | None = None
    next_id: str | None = None
    token_count: int | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for field_name in ("chunk_id", "document_id", "content", "embedding_text", "source_uri"):
            require_non_empty(getattr(self, field_name), field_name)
        require_non_negative(self.chunk_index, "chunk_index")
        require_range(self.page_start, self.page_end, "page_start", "page_end")
        require_range(self.char_start, self.char_end, "char_start", "char_end")
        require_non_negative(self.token_count, "token_count")
        object.__setattr__(self, "heading_path", tuple(self.heading_path))
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
