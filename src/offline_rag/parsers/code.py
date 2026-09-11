"""Parser for decoded source code with stable offsets and language metadata."""

from __future__ import annotations

import hashlib
from pathlib import Path

from offline_rag.contracts.documents import (
    ContentBlock,
    ContentBlockType,
    LoadedContent,
    ParsedDocument,
)
from offline_rag.exceptions import DocumentParseError
from offline_rag.normalization import normalize_text

CODE_LANGUAGES = {
    ".c": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cs": "csharp",
    ".go": "go",
    ".h": "c",
    ".hpp": "cpp",
    ".java": "java",
    ".js": "javascript",
    ".jsx": "javascript",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".php": "php",
    ".py": "python",
    ".rb": "ruby",
    ".rs": "rust",
    ".scala": "scala",
    ".sh": "shell",
    ".sql": "sql",
    ".swift": "swift",
    ".ts": "typescript",
    ".tsx": "typescript",
}

CODE_EXTENSIONS = tuple(CODE_LANGUAGES)


class CodeParser:
    name = "code"
    version = "1"

    def parse(self, loaded: LoadedContent) -> ParsedDocument:
        if loaded.text is None:
            raise DocumentParseError(
                "code parser requires decoded text",
                details={"source_id": loaded.source.source_id},
            )
        suffix = Path(loaded.source.relative_path or loaded.source.uri).suffix.lower()
        language = CODE_LANGUAGES.get(suffix)
        if language is None:
            raise DocumentParseError(
                "code parser does not recognize the source language",
                details={"source_id": loaded.source.source_id, "extension": suffix},
            )
        content = normalize_text(loaded.text, preserve_layout=True)
        if not content.strip():
            raise DocumentParseError(
                "source code document is empty",
                details={"source_id": loaded.source.source_id},
            )
        material = (
            f"{loaded.source.source_id}\0{loaded.source.content_hash}\0"
            f"{self.name}:{language}\0{self.version}"
        )
        return ParsedDocument(
            document_id=hashlib.sha256(material.encode()).hexdigest(),
            source=loaded.source,
            blocks=(
                ContentBlock(
                    ContentBlockType.CODE,
                    content,
                    char_start=0,
                    char_end=len(content),
                    metadata={"language": language},
                ),
            ),
            parser_name=f"{self.name}:{language}",
            parser_version=self.version,
            metadata={"format": "source_code", "language": language},
        )
