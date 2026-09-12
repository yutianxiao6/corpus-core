"""CSV, JSON and JSONL parsers preserving record boundaries."""

from __future__ import annotations

import csv
import hashlib
import io
import json
from collections.abc import Iterable

from corpuscore.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from corpuscore.exceptions import DocumentParseError


class StructuredTextParser:
    name = "structured_text"
    version = "1"

    def __init__(self, format_name: str) -> None:
        if format_name not in {"csv", "json", "jsonl"}:
            raise ValueError("format_name must be csv, json or jsonl")
        self.format_name = format_name

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.text is None:
            raise DocumentParseError(
                f"{self.format_name.upper()} parser requires decoded text",
                details={"source_id": loaded.source.source_id},
            )
        try:
            blocks = tuple(self._parse_content(loaded.text))
        except (csv.Error, json.JSONDecodeError, UnicodeError) as exc:
            raise DocumentParseError(
                f"cannot parse {self.format_name.upper()} source",
                details={"source_id": loaded.source.source_id},
            ) from exc
        if not blocks:
            raise DocumentParseError(
                f"{self.format_name.upper()} document has no records",
                details={"source_id": loaded.source.source_id},
            )
        material = (
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0"
            f"{self.name}:{self.format_name}\0{self.version}"
        )
        return ParsedDocument(
            document_id=hashlib.sha256(material.encode()).hexdigest(),
            source=loaded.source,
            blocks=blocks,
            parser_name=f"{self.name}:{self.format_name}",
            parser_version=self.version,
            metadata={"format": self.format_name, "record_count": len(blocks)},
        )

    def _parse_content(self, text: str) -> Iterable[ContentBlock]:
        if self.format_name == "csv":
            reader = csv.reader(io.StringIO(text))
            rows = list(reader)
            if not rows:
                return
            headers = rows[0]
            for row_number, row in enumerate(rows[1:], start=2):
                cells = [
                    f"{headers[index] if index < len(headers) else f'column_{index + 1}'}: {value}"
                    for index, value in enumerate(row)
                ]
                yield ContentBlock(
                    ContentBlockType.TABLE,
                    " | ".join(cells),
                    metadata={"row_number": row_number, "headers": tuple(headers)},
                )
            return
        if self.format_name == "json":
            value = json.loads(text)
            records = value if isinstance(value, list) else [value]
        else:
            records = [json.loads(line) for line in text.splitlines() if line.strip()]
        for index, record in enumerate(records):
            yield ContentBlock(
                ContentBlockType.METADATA,
                json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
                metadata={"record_index": index},
            )
