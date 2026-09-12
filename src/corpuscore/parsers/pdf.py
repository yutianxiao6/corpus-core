"""Text PDF parser preserving one-based page locations."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO

from pypdf import PdfReader

from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from corpuscore.exceptions import DocumentParseError
from corpuscore.normalization import normalize_text

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n+")


class PdfParser:
    name = "pdf"
    version = "1"

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.binary is None:
            raise DocumentParseError(
                "PDF parser requires binary content",
                details={"source_id": loaded.source.source_id},
            )
        try:
            reader = PdfReader(BytesIO(loaded.binary), strict=False)
            if reader.is_encrypted and reader.decrypt("") == 0:
                raise DocumentParseError("encrypted PDF requires a password")
            blocks: list[ContentBlock] = []
            for page_number, page in enumerate(reader.pages, start=1):
                page_text = normalize_text(page.extract_text() or "")
                for paragraph in _PARAGRAPH_BREAK.split(page_text):
                    content = paragraph.strip()
                    if content:
                        blocks.append(
                            ContentBlock(
                                block_type=ContentBlockType.PARAGRAPH,
                                content=content,
                                page=page_number,
                            )
                        )
                if page_number < len(reader.pages):
                    blocks.append(
                        ContentBlock(
                            block_type=ContentBlockType.PAGE_BREAK,
                            content="",
                            page=page_number,
                        )
                    )
            title = None
            if reader.metadata is not None and reader.metadata.title:
                title = str(reader.metadata.title).strip() or None
        except DocumentParseError:
            raise
        except Exception as exc:
            raise DocumentParseError(
                "cannot parse PDF source", details={"source_id": loaded.source.source_id}
            ) from exc
        if not any(block.block_type is not ContentBlockType.PAGE_BREAK for block in blocks):
            raise DocumentParseError(
                "PDF has no extractable text; enable the optional OCR pipeline",
                details={"source_id": loaded.source.source_id},
            )
        return ParsedDocument(
            document_id=self._document_id(loaded),
            source=loaded.source,
            title=title,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"format": "pdf", "page_count": len(reader.pages)},
        )

    def _document_id(self, loaded: LoadedContent) -> str:
        material = (
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0{self.name}\0{self.version}"
        )
        return hashlib.sha256(material.encode()).hexdigest()
