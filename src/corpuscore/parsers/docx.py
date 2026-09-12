"""DOCX parser preserving body order, headings, paragraphs and tables."""

from __future__ import annotations

import hashlib
import re
from io import BytesIO

from docx import Document
from docx.document import Document as DocumentObject
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from corpuscore.exceptions import DocumentParseError
from corpuscore.normalization import normalize_text

_HEADING_STYLE = re.compile(r"(?:heading|标题)\s*([1-6])", re.IGNORECASE)


class DocxParser:
    name = "docx"
    version = "1"

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.binary is None:
            raise DocumentParseError(
                "DOCX parser requires binary content",
                details={"source_id": loaded.source.source_id},
            )
        try:
            document = Document(BytesIO(loaded.binary))
            blocks = list(self._blocks(document))
            title = (document.core_properties.title or "").strip() or None
        except Exception as exc:
            raise DocumentParseError(
                "cannot parse DOCX source", details={"source_id": loaded.source.source_id}
            ) from exc
        if not blocks:
            raise DocumentParseError(
                "DOCX document has no indexable content",
                details={"source_id": loaded.source.source_id},
            )
        if title is None:
            title = next(
                (
                    block.content
                    for block in blocks
                    if block.block_type is ContentBlockType.HEADING and block.heading_level == 1
                ),
                None,
            )
        material = (
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0{self.name}\0{self.version}"
        )
        return ParsedDocument(
            document_id=hashlib.sha256(material.encode()).hexdigest(),
            source=loaded.source,
            title=title,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"format": "docx"},
        )

    def _blocks(self, document: DocumentObject):  # type: ignore[no-untyped-def]
        for child in document.element.body.iterchildren():
            if child.tag == qn("w:p"):
                paragraph = Paragraph(child, document)
                content = normalize_text(paragraph.text)
                if not content:
                    continue
                style_name = paragraph.style.name if paragraph.style is not None else ""
                heading = _HEADING_STYLE.search(style_name)
                block_type = ContentBlockType.HEADING if heading else ContentBlockType.PARAGRAPH
                if style_name.lower().startswith("list"):
                    block_type = ContentBlockType.LIST_ITEM
                yield ContentBlock(
                    block_type=block_type,
                    content=content,
                    heading_level=int(heading.group(1)) if heading else None,
                    metadata={"style": style_name} if style_name else {},
                )
            elif child.tag == qn("w:tbl"):
                table = Table(child, document)
                rows = [
                    " | ".join(normalize_text(cell.text) for cell in row.cells)
                    for row in table.rows
                ]
                content = "\n".join(row for row in rows if row.strip())
                if content:
                    yield ContentBlock(
                        block_type=ContentBlockType.TABLE,
                        content=content,
                        metadata={"row_count": len(rows)},
                    )
