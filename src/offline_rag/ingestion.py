"""Composable TXT/Markdown preview and indexing orchestration."""

from __future__ import annotations

import fnmatch
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from offline_rag.chunkers import (
    ChunkDeduplicator,
    ChunkFinalizer,
    ChunkSizeProcessor,
    HeadingContextInjector,
    HeadingRecursiveChunker,
    PageAwareChunker,
    ParagraphPackingChunker,
    ParentChildChunker,
    RecursiveChunker,
    TableRowChunker,
)
from offline_rag.config.models import (
    HeadingRecursiveChunkProfile,
    PageAwareChunkProfile,
    ParagraphPackingChunkProfile,
    ParentChildChunkProfile,
    RagConfig,
    RecursiveChunkProfile,
    TableRowsChunkProfile,
)
from offline_rag.contracts.chunks import Chunk
from offline_rag.contracts.common import JSONValue
from offline_rag.contracts.documents import ParsedDocument, SourceDescriptor
from offline_rag.contracts.indexing import (
    EmbeddingSpecification,
    IndexSpecification,
    IngestionItemResult,
    IngestionReport,
    IngestionStage,
    ItemFailure,
    VectorRecord,
)
from offline_rag.exceptions import ConfigurationError, OfflineRagError, UnsupportedDocumentError
from offline_rag.loaders import DocxLoader, MarkdownLoader, PdfLoader, TextLoader
from offline_rag.parsers import (
    DocxParser,
    HtmlParser,
    MarkdownParser,
    PdfParser,
    StructuredTextParser,
    TextParser,
)
from offline_rag.ports import EmbeddingProvider, VectorStorePort
from offline_rag.sources import (
    DiscoveryOptions,
    FileSystemSourceProvider,
    ManifestSourceProvider,
    apply_sidecar,
)
from offline_rag.vectorstores import chunk_to_payload

SUPPORTED_EXTENSIONS = (
    ".txt",
    ".md",
    ".markdown",
    ".pdf",
    ".docx",
    ".html",
    ".htm",
    ".csv",
    ".json",
    ".jsonl",
)


@dataclass(frozen=True, slots=True)
class PreviewItem:
    source: SourceDescriptor
    document: ParsedDocument
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class PreviewReport:
    items: tuple[PreviewItem, ...]
    failures: tuple[ItemFailure, ...]

    @property
    def source_count(self) -> int:
        return len(self.items) + len(self.failures)

    @property
    def chunk_count(self) -> int:
        return sum(len(item.chunks) for item in self.items)


class IngestionService:
    def __init__(
        self,
        config: RagConfig,
        *,
        embedding: EmbeddingProvider | None = None,
        vector_store: VectorStorePort | None = None,
        token_counter: Callable[[str], int] | None = None,
    ) -> None:
        self._config = config
        self._embedding = embedding
        self._vector_store = vector_store
        self._token_counter = token_counter
        self._text_loader = TextLoader()
        self._markdown_loader = MarkdownLoader()
        self._pdf_loader = PdfLoader()
        self._docx_loader = DocxLoader()
        self._structured_text_loader = TextLoader(
            extensions=(".html", ".htm", ".csv", ".json", ".jsonl"),
            media_types=("text/html", "text/csv", "application/json", "application/x-ndjson"),
        )
        self._text_parser = TextParser()
        self._markdown_parser = MarkdownParser()
        self._pdf_parser = PdfParser()
        self._docx_parser = DocxParser()
        self._html_parser = HtmlParser()
        self._finalizer = ChunkFinalizer()

    def preview(
        self, inputs: Sequence[str | Path], *, root: str | Path | None = None
    ) -> PreviewReport:
        items: list[PreviewItem] = []
        failures: list[ItemFailure] = []
        for source in self._sources(inputs, root=root):
            try:
                document = self._parse(source)
                chunks = self._chunk(document)
                items.append(PreviewItem(source, document, chunks))
            except Exception as exc:
                failure = self._failure(source.uri, IngestionStage.CHUNKED, exc)
                if not self._config.ingestion.continue_on_error:
                    raise
                failures.append(failure)
        return PreviewReport(tuple(items), tuple(failures))

    def build(
        self,
        inputs: Sequence[str | Path],
        specification: IndexSpecification,
        *,
        root: str | Path | None = None,
    ) -> IngestionReport:
        if self._embedding is None or self._vector_store is None:
            raise RuntimeError("build requires embedding and vector_store components")
        started = time.perf_counter()
        self._vector_store.ensure_index(specification)
        preview = self.preview(inputs, root=root)
        results: list[IngestionItemResult] = [
            IngestionItemResult(
                source_uri=failure.source_uri,
                stage=IngestionStage.FAILED,
                failure=failure,
            )
            for failure in preview.failures
        ]
        for item in preview.items:
            item_started = time.perf_counter()
            try:
                vectors = self._embedding.embed_documents(
                    [chunk.embedding_text for chunk in item.chunks]
                )
                if len(vectors) != len(item.chunks):
                    raise RuntimeError("embedding batch size does not match chunk count")
                records = [
                    VectorRecord(chunk.chunk_id, vector, chunk_to_payload(chunk))
                    for chunk, vector in zip(item.chunks, vectors, strict=True)
                ]
                batch_size = self._config.ingestion.commit_batch_size
                for offset in range(0, len(records), batch_size):
                    self._vector_store.upsert(records[offset : offset + batch_size])
                results.append(
                    IngestionItemResult(
                        source_uri=item.source.uri,
                        stage=IngestionStage.INDEXED,
                        chunk_count=len(item.chunks),
                        duration_ms=(time.perf_counter() - item_started) * 1000,
                    )
                )
            except Exception as exc:
                failure = self._failure(item.source.uri, IngestionStage.EMBEDDED, exc)
                if not self._config.ingestion.continue_on_error:
                    raise
                results.append(
                    IngestionItemResult(
                        source_uri=item.source.uri,
                        stage=IngestionStage.FAILED,
                        duration_ms=(time.perf_counter() - item_started) * 1000,
                        failure=failure,
                    )
                )
        return IngestionReport(
            job_id=f"build-{time.time_ns()}",
            index_version=specification.fingerprint()[:16],
            items=results,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def _sources(
        self, inputs: Sequence[str | Path], *, root: str | Path | None
    ) -> Sequence[SourceDescriptor]:
        ingestion = self._config.ingestion
        options = DiscoveryOptions(
            recursive=ingestion.recursive,
            follow_symlinks=ingestion.follow_symlinks,
            include_hidden=ingestion.include_hidden,
            ignore_patterns=ingestion.ignore_patterns,
            allowed_extensions=SUPPORTED_EXTENSIONS,
        )
        if len(inputs) == 1 and Path(inputs[0]).suffix.lower() in (".yaml", ".yml"):
            sources = ManifestSourceProvider(inputs[0], options=options).discover()
        else:
            sources = FileSystemSourceProvider(inputs, root=root, options=options).discover()
        return tuple(apply_sidecar(source) for source in sources)

    def _parse(self, source: SourceDescriptor) -> ParsedDocument:
        suffix = Path(source.relative_path or source.uri).suffix.lower()
        if suffix == ".txt":
            return self._text_parser.parse(self._text_loader.load(source))
        if suffix in (".md", ".markdown"):
            return self._markdown_parser.parse(self._markdown_loader.load(source))
        if suffix == ".pdf":
            return self._pdf_parser.parse(self._pdf_loader.load(source))
        if suffix == ".docx":
            return self._docx_parser.parse(self._docx_loader.load(source))
        if suffix in (".html", ".htm"):
            return self._html_parser.parse(self._structured_text_loader.load(source))
        if suffix in (".csv", ".json", ".jsonl"):
            return StructuredTextParser(suffix.removeprefix(".")).parse(
                self._structured_text_loader.load(source)
            )
        raise UnsupportedDocumentError(
            "no built-in parser supports this source",
            details={"source_id": source.source_id, "extension": suffix},
        )

    def _chunk(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        name = self._profile_for(document.source)
        profile = self._config.chunk_profiles[name]
        if isinstance(profile, RecursiveChunkProfile):
            if profile.length_unit == "token" and self._token_counter is None:
                raise ConfigurationError(
                    "token length_unit requires the configured local model tokenizer"
                )
            length_function = self._token_counter if profile.length_unit == "token" else len
            assert length_function is not None
            chunker = RecursiveChunker(
                chunk_size=profile.chunk_size,
                chunk_overlap=profile.chunk_overlap,
                length_function=length_function,
                count_tokens=profile.length_unit == "token",
            )
            drafts = chunker.split(document)
            drafts = ChunkSizeProcessor(
                minimum_size=profile.minimum_size,
                maximum_size=profile.chunk_size,
                length_function=length_function,
                count_tokens=profile.length_unit == "token",
            ).process(drafts)
        elif isinstance(profile, HeadingRecursiveChunkProfile):
            chunker = HeadingRecursiveChunker(
                max_chunk_size=profile.max_chunk_size,
                chunk_overlap=profile.chunk_overlap,
            )
            drafts = chunker.split(document)
        elif isinstance(profile, PageAwareChunkProfile):
            drafts = PageAwareChunker(
                chunk_size=profile.chunk_size, chunk_overlap=profile.chunk_overlap
            ).split(document)
        elif isinstance(profile, ParagraphPackingChunkProfile):
            drafts = ParagraphPackingChunker(
                target_size=profile.target_size, maximum_size=profile.maximum_size
            ).split(document)
        elif isinstance(profile, TableRowsChunkProfile):
            drafts = TableRowChunker(
                max_rows_per_chunk=profile.max_rows_per_chunk,
                repeat_headers=profile.repeat_headers,
            ).split(document)
        elif isinstance(profile, ParentChildChunkProfile):
            drafts = ParentChildChunker(
                parent_size=profile.parent_size,
                child_size=profile.child_size,
                child_overlap=profile.child_overlap,
            ).split(document)
        else:
            raise UnsupportedDocumentError(
                f"chunk profile type is not implemented in P1: {profile.type}"
            )
        if getattr(profile, "include_heading_in_embedding", True):
            drafts = HeadingContextInjector().process(drafts)
        drafts = ChunkDeduplicator().process(drafts)
        return self._finalizer.finalize(drafts)

    def _profile_for(self, source: SourceDescriptor) -> str:
        relative_path = source.relative_path or Path(source.uri).name
        extension = Path(relative_path).suffix.lower()
        explicit_profile = source.metadata.get("rag_profile")
        if isinstance(explicit_profile, str):
            if explicit_profile not in self._config.chunk_profiles:
                raise ConfigurationError(
                    f"source references unknown chunk profile: {explicit_profile}"
                )
            return explicit_profile
        for rule in self._config.routing:
            extension_match = not rule.match.extensions or extension in {
                value.lower() for value in rule.match.extensions
            }
            filename_match = not rule.match.filename_patterns or any(
                fnmatch.fnmatchcase(Path(relative_path).name, pattern)
                for pattern in rule.match.filename_patterns
            )
            path_match = not rule.match.path_patterns or any(
                fnmatch.fnmatchcase(relative_path, pattern) for pattern in rule.match.path_patterns
            )
            metadata_match = all(
                source.metadata.get(key) == value for key, value in rule.match.metadata.items()
            )
            if extension_match and filename_match and path_match and metadata_match:
                return rule.use
        if extension in (".md", ".markdown") and "markdown_heading" in self._config.chunk_profiles:
            return "markdown_heading"
        return "default"

    @staticmethod
    def _failure(source_uri: str, stage: IngestionStage, exc: Exception) -> ItemFailure:
        if isinstance(exc, OfflineRagError):
            code = exc.code
            message = str(exc)
        else:
            code = "unexpected_error"
            message = type(exc).__name__
        return ItemFailure(source_uri=source_uri, stage=stage, error_code=code, message=message)


def build_index_specification(
    config: RagConfig, embedding: EmbeddingSpecification
) -> IndexSpecification:
    dumped = config.model_dump(mode="json")["chunk_profiles"]
    chunking = cast(Mapping[str, JSONValue], dumped)
    return IndexSpecification(
        index_format_version=1,
        payload_schema_version=1,
        embedding=embedding,
        parser_versions={
            "text": TextParser.version,
            "markdown": MarkdownParser.version,
            "pdf": PdfParser.version,
            "docx": DocxParser.version,
            "html": HtmlParser.version,
            "structured_text": StructuredTextParser.version,
        },
        chunking_configuration=chunking,
    )
