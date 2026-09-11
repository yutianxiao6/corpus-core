"""Offline HTML structure parser with no external resource loading."""

from __future__ import annotations

import hashlib

from bs4 import BeautifulSoup, Tag

from offline_rag.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from offline_rag.exceptions import DocumentParseError
from offline_rag.normalization import normalize_text

_STRUCTURAL_TAGS = {"h1", "h2", "h3", "h4", "h5", "h6", "p", "li", "pre", "table"}


class HtmlParser:
    name = "html"
    version = "1"

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.text is None:
            raise DocumentParseError(
                "HTML parser requires decoded text",
                details={"source_id": loaded.source.source_id},
            )
        try:
            soup = BeautifulSoup(loaded.text, "lxml")
            for element in soup(
                ["script", "style", "noscript", "template", "nav", "header", "footer", "aside"]
            ):
                element.decompose()
            title = normalize_text(soup.title.get_text(" ")) if soup.title else None
            root = soup.find("main") or soup.find("article") or soup.body or soup
            blocks = [
                block for tag in root.find_all(_STRUCTURAL_TAGS) if (block := self._block(tag))
            ]
        except Exception as exc:
            raise DocumentParseError(
                "cannot parse HTML source", details={"source_id": loaded.source.source_id}
            ) from exc
        if not blocks:
            raise DocumentParseError(
                "HTML document has no indexable content",
                details={"source_id": loaded.source.source_id},
            )
        material = (
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0{self.name}\0{self.version}"
        )
        return ParsedDocument(
            document_id=hashlib.sha256(material.encode()).hexdigest(),
            source=loaded.source,
            title=title or None,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"format": "html"},
        )

    @staticmethod
    def _block(tag: Tag) -> ContentBlock | None:
        if any(
            isinstance(parent, Tag) and parent.name in _STRUCTURAL_TAGS for parent in tag.parents
        ):
            return None
        if tag.name == "table":
            rows = [
                " | ".join(
                    normalize_text(cell.get_text(" ")) for cell in row.find_all(["th", "td"])
                )
                for row in tag.find_all("tr")
            ]
            content = "\n".join(row for row in rows if row)
            block_type = ContentBlockType.TABLE
        else:
            separator = "\n" if tag.name == "pre" else " "
            content = normalize_text(
                tag.get_text(separator), preserve_layout=tag.name == "pre"
            ).strip()
            if tag.name and tag.name.startswith("h"):
                block_type = ContentBlockType.HEADING
            elif tag.name == "li":
                block_type = ContentBlockType.LIST_ITEM
            elif tag.name == "pre":
                block_type = ContentBlockType.CODE
            else:
                block_type = ContentBlockType.PARAGRAPH
        if not content:
            return None
        heading_level = int(tag.name[1]) if tag.name and tag.name.startswith("h") else None
        return ContentBlock(
            block_type=block_type,
            content=content,
            heading_level=heading_level,
            metadata={"html_tag": tag.name or ""},
        )
