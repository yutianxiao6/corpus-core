"""Canonical source and parsed-document contracts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from corpuscore.contracts.common import (
    JSONValue,
    freeze_metadata,
    require_non_empty,
    require_non_negative,
    require_range,
)


class ContentBlockType(StrEnum):
    HEADING = "heading"
    PARAGRAPH = "paragraph"
    LIST_ITEM = "list_item"
    CODE = "code"
    TABLE = "table"
    PAGE_BREAK = "page_break"
    IMAGE_TEXT = "image_text"
    METADATA = "metadata"


@dataclass(frozen=True, slots=True)
class LoadedContent:
    """Content loaded from a source before structure-aware parsing."""

    source: SourceDescriptor
    text: str | None = None
    binary: bytes | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.text is None and self.binary is None:
            raise ValueError("loaded content must contain text or binary data")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class SourceDescriptor:
    source_id: str
    uri: str
    content_hash: str
    relative_path: str | None = None
    media_type: str | None = None
    size_bytes: int | None = None
    modified_at: datetime | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.source_id, "source_id")
        require_non_empty(self.uri, "uri")
        require_non_empty(self.content_hash, "content_hash")
        require_non_negative(self.size_bytes, "size_bytes")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ContentBlock:
    block_type: ContentBlockType
    content: str
    page: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    heading_level: int | None = None
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.block_type is not ContentBlockType.PAGE_BREAK:
            require_non_empty(self.content, "content")
        require_non_negative(self.page, "page")
        require_range(self.char_start, self.char_end, "char_start", "char_end")
        if self.heading_level is not None and self.heading_level <= 0:
            raise ValueError("heading_level must be positive")
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ParsedDocument:
    document_id: str
    source: SourceDescriptor
    blocks: Sequence[ContentBlock]
    parser_name: str
    parser_version: str
    title: str | None = None
    language_hints: Sequence[str] = ()
    metadata: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        require_non_empty(self.document_id, "document_id")
        require_non_empty(self.parser_name, "parser_name")
        require_non_empty(self.parser_version, "parser_version")
        if not self.blocks:
            raise ValueError("blocks must contain at least one content block")
        languages = tuple(language.strip() for language in self.language_hints if language.strip())
        object.__setattr__(self, "blocks", tuple(self.blocks))
        object.__setattr__(self, "language_hints", languages)
        object.__setattr__(self, "metadata", freeze_metadata(self.metadata))
