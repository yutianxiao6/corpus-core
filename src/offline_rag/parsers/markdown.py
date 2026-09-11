"""Deterministic Markdown block parser with source offsets."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from offline_rag.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from offline_rag.exceptions import DocumentParseError
from offline_rag.normalization import DocumentNormalizer

_HEADING = re.compile(r"^[ \t]{0,3}(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*(?:\n)?$")
_FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})[ \t]*([^\n`]*)?(?:\n)?$")
_LIST_ITEM = re.compile(r"^[ \t]*(?:[-+*]|\d+[.)])[ \t]+(.+?)[ \t]*(?:\n)?$")


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    start: int
    end: int


class MarkdownParser:
    name = "markdown"
    version = "1"

    def __init__(self, *, normalize: bool = True) -> None:
        self._normalizer = DocumentNormalizer() if normalize else None

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.text is None:
            raise DocumentParseError(
                "Markdown parser requires decoded text",
                details={"source_id": loaded.source.source_id},
            )
        if self._normalizer is not None:
            loaded = self._normalizer.normalize_loaded(loaded)
        text = loaded.text
        if text is None or not text.strip():
            raise DocumentParseError(
                "Markdown document is empty",
                details={"source_id": loaded.source.source_id},
            )

        lines = self._lines(text)
        blocks: list[ContentBlock] = []
        title: str | None = None
        index = 0
        while index < len(lines):
            line = lines[index]
            if not line.text.strip():
                index += 1
                continue
            heading = _HEADING.match(line.text)
            if heading:
                content = heading.group(2).strip()
                content_start = line.start + line.text.find(heading.group(2))
                blocks.append(
                    ContentBlock(
                        block_type=ContentBlockType.HEADING,
                        content=content,
                        char_start=content_start,
                        char_end=content_start + len(content),
                        heading_level=len(heading.group(1)),
                    )
                )
                if title is None and len(heading.group(1)) == 1:
                    title = content
                index += 1
                continue
            fence = _FENCE.match(line.text)
            if fence:
                block, index = self._parse_fence(lines, index, fence)
                if block is not None:
                    blocks.append(block)
                continue
            list_item = _LIST_ITEM.match(line.text)
            if list_item:
                content = list_item.group(1).strip()
                content_start = line.start + line.text.find(list_item.group(1))
                blocks.append(
                    ContentBlock(
                        block_type=ContentBlockType.LIST_ITEM,
                        content=content,
                        char_start=content_start,
                        char_end=content_start + len(content),
                    )
                )
                index += 1
                continue
            block, index = self._parse_paragraph(lines, index)
            blocks.append(block)

        if not blocks:
            raise DocumentParseError(
                "Markdown document does not contain indexable content",
                details={"source_id": loaded.source.source_id},
            )
        document_id = hashlib.sha256(
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0{self.name}\0{self.version}".encode()
        ).hexdigest()
        document = ParsedDocument(
            document_id=document_id,
            source=loaded.source,
            title=title,
            blocks=blocks,
            parser_name=self.name,
            parser_version=self.version,
            metadata={"format": "markdown"},
        )
        if self._normalizer is not None:
            document = self._normalizer.normalize_document(document)
        return document

    @staticmethod
    def _lines(text: str) -> list[_Line]:
        result: list[_Line] = []
        offset = 0
        for value in text.splitlines(keepends=True):
            result.append(_Line(value, offset, offset + len(value)))
            offset += len(value)
        if offset < len(text):
            result.append(_Line(text[offset:], offset, len(text)))
        return result

    @staticmethod
    def _parse_fence(
        lines: list[_Line], index: int, opening: re.Match[str]
    ) -> tuple[ContentBlock | None, int]:
        marker = opening.group(1)
        language = (opening.group(2) or "").strip()
        content_start = lines[index].end
        cursor = index + 1
        while cursor < len(lines) and not MarkdownParser._is_closing_fence(
            lines[cursor].text, marker
        ):
            cursor += 1
        has_closing_fence = cursor < len(lines)
        content_end = lines[cursor].start if has_closing_fence else lines[-1].end
        content = "".join(line.text for line in lines[index + 1 : cursor])
        content = content.removesuffix("\n")
        next_index = cursor + 1 if has_closing_fence else len(lines)
        if not content:
            return None, next_index
        return (
            ContentBlock(
                block_type=ContentBlockType.CODE,
                content=content,
                char_start=content_start,
                char_end=content_end,
                metadata={"language": language} if language else {},
            ),
            next_index,
        )

    @staticmethod
    def _is_closing_fence(line: str, opening_marker: str) -> bool:
        stripped = line.strip()
        return len(stripped) >= len(opening_marker) and set(stripped) == {opening_marker[0]}

    @staticmethod
    def _parse_paragraph(lines: list[_Line], index: int) -> tuple[ContentBlock, int]:
        cursor = index + 1
        while cursor < len(lines):
            text = lines[cursor].text
            if (
                not text.strip()
                or _HEADING.match(text)
                or _FENCE.match(text)
                or _LIST_ITEM.match(text)
            ):
                break
            cursor += 1
        raw = "".join(line.text for line in lines[index:cursor])
        content = raw.strip()
        leading = len(raw) - len(raw.lstrip())
        start = lines[index].start + leading
        return (
            ContentBlock(
                block_type=ContentBlockType.PARAGRAPH,
                content=content,
                char_start=start,
                char_end=start + len(content),
            ),
            cursor,
        )
