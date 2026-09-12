"""LangChain BaseRetriever adapter for the native retrieval engine."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from langchain_core.callbacks import (
    AsyncCallbackManagerForRetrieverRun,
    CallbackManagerForRetrieverRun,
)
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import Field, field_validator

from corpuscore.contracts.retrieval import QueryOverrides, RetrievalResult
from corpuscore.engine import RetrievalEngine


class CorpusCoreLangChainRetriever(BaseRetriever):  # type: ignore[misc]
    """Expose native retrieval through LangChain's standard Runnable interface."""

    engine: RetrievalEngine = Field(exclude=True)
    profile: str = "fast"
    overrides: object = Field(default_factory=QueryOverrides)

    @field_validator("overrides")
    @classmethod
    def _validate_overrides(cls, value: object) -> object:
        if not isinstance(value, QueryOverrides):
            raise TypeError("overrides must be a QueryOverrides instance")
        return value

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        result = self.engine.query(
            query,
            profile=self.profile,
            overrides=cast(QueryOverrides, self.overrides),
        )
        return self._documents(result)

    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: AsyncCallbackManagerForRetrieverRun,
    ) -> list[Document]:
        del run_manager
        result = await self.engine.aquery(
            query,
            profile=self.profile,
            overrides=cast(QueryOverrides, self.overrides),
        )
        return self._documents(result)

    @staticmethod
    def _documents(result: RetrievalResult) -> list[Document]:
        documents: list[Document] = []
        for position, candidate in enumerate(result.hits):
            chunk = candidate.chunk
            citation = result.citations[position] if position < len(result.citations) else None
            metadata = {
                "chunk_id": chunk.chunk_id,
                "document_id": chunk.document_id,
                "source": chunk.source_uri,
                "source_uri": chunk.source_uri,
                "chunk_index": chunk.chunk_index,
                "title": chunk.title,
                "heading_path": list(chunk.heading_path),
                "page_start": chunk.page_start,
                "page_end": chunk.page_end,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
                "parent_id": chunk.parent_id,
                "previous_id": chunk.previous_id,
                "next_id": chunk.next_id,
                "dense_score": candidate.dense_score,
                "sparse_score": candidate.sparse_score,
                "fusion_score": candidate.fusion_score,
                "rerank_score": candidate.rerank_score,
                "final_score": candidate.final_score,
                "rank": candidate.rank,
                "origins": list(candidate.origins),
                "index_version": result.index_version,
                "embedding_fingerprint": result.embedding_fingerprint,
                "source_metadata": _plain(chunk.metadata),
            }
            if citation is not None:
                metadata["citation"] = {
                    "id": citation.citation_id,
                    "chunk_ids": list(citation.chunk_ids),
                    "source_uri": citation.source_uri,
                    "page_start": citation.page_start,
                    "page_end": citation.page_end,
                    "title": citation.title,
                }
            documents.append(
                Document(id=chunk.chunk_id, page_content=chunk.content, metadata=metadata)
            )
        return documents


def _plain(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value
