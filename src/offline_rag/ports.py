"""Stable component protocols implemented by built-ins and plugins."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterable, Sequence
from typing import Protocol, TypeAlias

from offline_rag.contracts.chunks import ChunkDraft
from offline_rag.contracts.documents import LoadedContent, ParsedDocument, SourceDescriptor
from offline_rag.contracts.indexing import (
    DeleteReport,
    EmbeddingSpecification,
    IndexSpecification,
    SparseEmbeddingSpecification,
    UpsertReport,
    VectorRecord,
)
from offline_rag.contracts.retrieval import (
    ContextBudget,
    OrganizedResult,
    RetrievalCandidate,
    RetrievalRequest,
    SearchHit,
    SearchRequest,
)

Vector: TypeAlias = Sequence[float]
SparseVector: TypeAlias = tuple[Sequence[int], Sequence[float]]


class SourceProvider(Protocol):
    def discover(self) -> Iterable[SourceDescriptor]: ...


class DocumentLoader(Protocol):
    def supports(self, source: SourceDescriptor) -> bool: ...

    def load(self, source: SourceDescriptor) -> LoadedContent: ...


class DocumentParser(Protocol):
    def parse(self, loaded: LoadedContent) -> ParsedDocument: ...


class Chunker(Protocol):
    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]: ...


class ChunkProcessor(Protocol):
    def process(self, chunks: Sequence[ChunkDraft]) -> Sequence[ChunkDraft]: ...


class EmbeddingProvider(Protocol):
    @property
    def specification(self) -> EmbeddingSpecification: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...

    async def aembed_query(self, text: str) -> Vector: ...


class SparseEmbeddingProvider(Protocol):
    @property
    def specification(self) -> SparseEmbeddingSpecification: ...

    def embed_documents(self, texts: Sequence[str]) -> Sequence[SparseVector]: ...

    def embed_query(self, text: str) -> SparseVector: ...

    async def aembed_query(self, text: str) -> SparseVector: ...


class VectorStorePort(Protocol):
    def ensure_index(self, specification: IndexSpecification) -> None: ...

    def upsert(self, records: Sequence[VectorRecord]) -> UpsertReport: ...

    def delete(self, chunk_ids: Sequence[str]) -> DeleteReport: ...

    def search(self, request: SearchRequest) -> Sequence[SearchHit]: ...

    async def asearch(self, request: SearchRequest) -> Sequence[SearchHit]: ...


class RetrievalStrategy(Protocol):
    def retrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]: ...

    async def aretrieve(self, request: RetrievalRequest) -> Sequence[RetrievalCandidate]: ...


class ResultOrganizer(Protocol):
    def organize(
        self,
        query: str,
        candidates: Sequence[RetrievalCandidate],
        budget: ContextBudget,
    ) -> OrganizedResult: ...


class IngestionEventStream(Protocol):
    def __aiter__(self) -> AsyncIterator[object]: ...
