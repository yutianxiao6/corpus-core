"""Python AST-aware chunking with deterministic fallback for other code."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

from corpuscore.contracts.chunks import ChunkDraft
from corpuscore.contracts.documents import ContentBlockType, ParsedDocument
from corpuscore.ports import Chunker

_DefinitionNode = ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef


@dataclass(frozen=True, slots=True)
class _Region:
    start: int
    end: int
    symbol: str
    node_type: str


class SyntaxChunker:
    """Keep top-level Python functions/classes intact when size permits."""

    name = "syntax"
    version = "1"

    def __init__(self, fallback: Chunker, *, max_chunk_size: int = 1600) -> None:
        if max_chunk_size <= 0:
            raise ValueError("max_chunk_size must be positive")
        self.fallback = fallback
        self.max_chunk_size = max_chunk_size

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        language = str(document.metadata.get("language", ""))
        suffix = Path(document.source.relative_path or document.source.uri).suffix.lower()
        if language != "python" and suffix != ".py":
            return self._fallback(document, "unsupported_language")
        code = "\n\n".join(
            block.content
            for block in document.blocks
            if block.block_type is not ContentBlockType.PAGE_BREAK
        )
        try:
            tree = ast.parse(code)
        except (SyntaxError, ValueError, TypeError):
            return self._fallback(document, "invalid_python_syntax")
        regions = self._regions(code, tree)
        if not regions:
            return self._fallback(document, "no_top_level_symbols")
        return self._drafts(document, code, regions)

    def _drafts(
        self, document: ParsedDocument, code: str, regions: Sequence[_Region]
    ) -> tuple[ChunkDraft, ...]:
        drafts: list[ChunkDraft] = []
        group: list[_Region] = []
        for region in regions:
            if region.end - region.start > self.max_chunk_size:
                if group:
                    drafts.append(self._group_draft(document, code, group))
                    group = []
                drafts.extend(self._split_oversize(document, code, region))
                continue
            proposed_start = group[0].start if group else region.start
            if group and region.end - proposed_start > self.max_chunk_size:
                drafts.append(self._group_draft(document, code, group))
                group = []
            group.append(region)
        if group:
            drafts.append(self._group_draft(document, code, group))
        return tuple(drafts)

    def _group_draft(
        self, document: ParsedDocument, code: str, regions: Sequence[_Region]
    ) -> ChunkDraft:
        start, end = self._trim(code, regions[0].start, regions[-1].end)
        return ChunkDraft(
            document_id=document.document_id,
            content=code[start:end],
            source_uri=document.source.uri,
            title=document.title,
            char_start=start,
            char_end=end,
            metadata={
                "chunker": self.name,
                "chunker_version": self.version,
                "language": "python",
                "syntax_symbols": tuple(region.symbol for region in regions),
                "syntax_node_types": tuple(region.node_type for region in regions),
            },
        )

    def _split_oversize(
        self, document: ParsedDocument, code: str, region: _Region
    ) -> tuple[ChunkDraft, ...]:
        content = code[region.start : region.end]
        drafts: list[ChunkDraft] = []
        offset = 0
        while offset < len(content):
            end = min(len(content), offset + self.max_chunk_size)
            if end < len(content):
                newline = content.rfind("\n", offset, end)
                if newline > offset:
                    end = newline + 1
            local_start, local_end = self._trim(content, offset, end)
            if local_start < local_end:
                drafts.append(
                    ChunkDraft(
                        document_id=document.document_id,
                        content=content[local_start:local_end],
                        source_uri=document.source.uri,
                        title=document.title,
                        char_start=region.start + local_start,
                        char_end=region.start + local_end,
                        metadata={
                            "chunker": self.name,
                            "chunker_version": self.version,
                            "language": "python",
                            "syntax_symbols": (region.symbol,),
                            "syntax_node_types": (region.node_type,),
                            "syntax_oversize_split": True,
                        },
                    )
                )
            offset = end
        return tuple(drafts)

    def _fallback(self, document: ParsedDocument, reason: str) -> tuple[ChunkDraft, ...]:
        return tuple(
            replace(
                draft,
                metadata={
                    **draft.metadata,
                    "syntax_fallback": True,
                    "syntax_fallback_reason": reason,
                },
            )
            for draft in self.fallback.split(document)
        )

    @staticmethod
    def _regions(code: str, tree: ast.Module) -> tuple[_Region, ...]:
        lines = code.splitlines(keepends=True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))
        definitions: list[tuple[int, _DefinitionNode]] = []
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                decorator_lines = [decorator.lineno for decorator in node.decorator_list]
                start_line = min([node.lineno, *decorator_lines])
                definitions.append((offsets[start_line - 1], node))
        if not definitions:
            return ()
        regions: list[_Region] = []
        first_start = definitions[0][0]
        if code[:first_start].strip():
            regions.append(_Region(0, first_start, "<module>", "ModulePreamble"))
        for index, (start, definition) in enumerate(definitions):
            end = definitions[index + 1][0] if index + 1 < len(definitions) else len(code)
            regions.append(_Region(start, end, definition.name, type(definition).__name__))
        return tuple(regions)

    @staticmethod
    def _trim(text: str, start: int, end: int) -> tuple[int, int]:
        while start < end and text[start].isspace():
            start += 1
        while end > start and text[end - 1].isspace():
            end -= 1
        return start, end
