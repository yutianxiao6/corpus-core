"""Conservative Unicode and whitespace normalization."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from offline_rag.contracts.documents import ContentBlockType, LoadedContent, ParsedDocument

_HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")
_EXCESS_BLANK_LINES = re.compile(r"\n{3,}")


def normalize_text(text: str, *, preserve_layout: bool = False) -> str:
    """Normalize text without changing code/table layout unless explicitly allowed."""

    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = "".join(
        character
        for character in normalized
        if character in ("\n", "\t") or unicodedata.category(character) != "Cc"
    )
    if preserve_layout:
        return normalized
    lines = (_HORIZONTAL_WHITESPACE.sub(" ", line).strip() for line in normalized.split("\n"))
    return _EXCESS_BLANK_LINES.sub("\n\n", "\n".join(lines)).strip()


class DocumentNormalizer:
    """Normalize parsed block content while preserving code and table layout."""

    def normalize_loaded(self, loaded: LoadedContent) -> LoadedContent:
        if loaded.text is None:
            return loaded
        metadata = dict(loaded.metadata)
        metadata["unicode_normalization"] = "NFC"
        return replace(
            loaded,
            text=normalize_text(loaded.text, preserve_layout=True),
            metadata=metadata,
        )

    def normalize_document(self, document: ParsedDocument) -> ParsedDocument:
        blocks = []
        for block in document.blocks:
            preserve_layout = block.block_type in (ContentBlockType.CODE, ContentBlockType.TABLE)
            content = normalize_text(block.content, preserve_layout=preserve_layout)
            if content or block.block_type is ContentBlockType.PAGE_BREAK:
                blocks.append(replace(block, content=content))
        metadata = dict(document.metadata)
        metadata["normalized"] = True
        return replace(document, blocks=blocks, metadata=metadata)
