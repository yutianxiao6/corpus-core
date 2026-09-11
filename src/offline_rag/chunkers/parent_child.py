"""Parent-child chunking with parent content retained in each child payload."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import replace

from offline_rag.chunkers.recursive import RecursiveChunker
from offline_rag.contracts.chunks import ChunkDraft
from offline_rag.contracts.documents import ParsedDocument


class ParentChildChunker:
    name = "parent_child"
    version = "1"

    def __init__(
        self,
        *,
        parent_size: int = 1400,
        child_size: int = 350,
        child_overlap: int = 60,
    ) -> None:
        if child_size >= parent_size:
            raise ValueError("child_size must be smaller than parent_size")
        self._parents = RecursiveChunker(chunk_size=parent_size, chunk_overlap=0)
        self._children = RecursiveChunker(chunk_size=child_size, chunk_overlap=child_overlap)

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        children: list[ChunkDraft] = []
        for parent_index, parent in enumerate(self._parents.split(document)):
            parent_key = hashlib.sha256(
                f"{document.document_id}\0{parent_index}\0{parent.content}".encode()
            ).hexdigest()
            for child_content, offset in self._children.split_text(parent.content):
                metadata = dict(parent.metadata)
                metadata.update(
                    {
                        "chunker": self.name,
                        "chunker_version": self.version,
                        "record_type": "child",
                        "parent_content": parent.content,
                    }
                )
                char_start = parent.char_start + offset if parent.char_start is not None else None
                children.append(
                    replace(
                        parent,
                        content=child_content,
                        embedding_text=None,
                        char_start=char_start,
                        char_end=(
                            char_start + len(child_content) if char_start is not None else None
                        ),
                        parent_key=parent_key,
                        token_count=None,
                        metadata=metadata,
                    )
                )
        return tuple(children)
