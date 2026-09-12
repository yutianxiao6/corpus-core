"""Stable chunk IDs and document-local neighbor relationships."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from corpuscore.contracts.chunks import Chunk, ChunkDraft
from corpuscore.exceptions import ChunkingError


class ChunkFinalizer:
    version = "1"

    def finalize(self, drafts: Sequence[ChunkDraft]) -> tuple[Chunk, ...]:
        if not drafts:
            return ()
        document_ids = {draft.document_id for draft in drafts}
        if len(document_ids) != 1:
            raise ChunkingError("chunk finalization requires drafts from exactly one document")
        chunk_ids = [self._chunk_id(draft, index) for index, draft in enumerate(drafts)]
        result: list[Chunk] = []
        for index, draft in enumerate(drafts):
            metadata = dict(draft.metadata)
            metadata["content_hash"] = hashlib.sha256(draft.content.encode()).hexdigest()
            metadata["finalizer_version"] = self.version
            parent_id = (
                self._parent_id(draft.document_id, draft.parent_key) if draft.parent_key else None
            )
            result.append(
                Chunk(
                    chunk_id=chunk_ids[index],
                    document_id=draft.document_id,
                    content=draft.content,
                    embedding_text=draft.text_for_embedding,
                    source_uri=draft.source_uri,
                    chunk_index=index,
                    title=draft.title,
                    heading_path=draft.heading_path,
                    page_start=draft.page_start,
                    page_end=draft.page_end,
                    char_start=draft.char_start,
                    char_end=draft.char_end,
                    parent_id=parent_id,
                    previous_id=chunk_ids[index - 1] if index > 0 else None,
                    next_id=chunk_ids[index + 1] if index + 1 < len(chunk_ids) else None,
                    token_count=draft.token_count,
                    metadata=metadata,
                )
            )
        return tuple(result)

    @classmethod
    def _chunk_id(cls, draft: ChunkDraft, index: int) -> str:
        material = f"{cls.version}\0{draft.document_id}\0{index}\0{draft.content}".encode()
        return hashlib.sha256(material).hexdigest()

    @classmethod
    def _parent_id(cls, document_id: str, parent_key: str) -> str:
        material = f"{cls.version}\0{document_id}\0parent\0{parent_key}".encode()
        return hashlib.sha256(material).hexdigest()
