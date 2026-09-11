"""Paragraph-preserving parser for decoded plain text."""

from __future__ import annotations

import hashlib
import re

from offline_rag.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from offline_rag.exceptions import DocumentParseError
from offline_rag.normalization import DocumentNormalizer, normalize_text

_PARAGRAPH = re.compile(r"\S(?:.*?\S)?(?=\n\s*\n|\Z)", re.DOTALL)


class TextParser:
    name = "text"
    version = "1"

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalize = normalize

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.text is None:
            raise DocumentParseError(
                "text parser requires decoded text",
                details={"source_id": loaded.source.source_id},
            )
        text = normalize_text(loaded.text, preserve_layout=True) if self._normalize else loaded.text
        blocks = [
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=match.group(0),
                char_start=match.start(),
                char_end=match.end(),
            )
            for match in _PARAGRAPH.finditer(text)
        ]
        if not blocks:
            raise DocumentParseError(
                "text document is empty", details={"source_id": loaded.source.source_id}
            )
        document_id = hashlib.sha256(
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0{self.name}\0{self.version}".encode()
        ).hexdigest()
        document = ParsedDocument(
            document_id=document_id,
            source=loaded.source,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"format": "text"},
        )
        return DocumentNormalizer().normalize_document(document) if self._normalize else document
