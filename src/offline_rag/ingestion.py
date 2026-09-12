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
    SemanticChunker,
    SyntaxChunker,
    TableRowChunker,
)
from offline_rag.config.models import (
    HeadingRecursiveChunkProfile,
    PageAwareChunkProfile,
    ParagraphPackingChunkProfile,
    ParentChildChunkProfile,
    RagConfig,
    RecursiveChunkProfile,
    SemanticChunkProfile,
    SyntaxChunkProfile,
    TableRowsChunkProfile,
)
from offline_rag.contracts.chunks import Chunk, ChunkDraft
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
from offline_rag.indexing import IndexedSource, IngestionJournal, plan_sync
from offline_rag.loaders import (
    DocxLoader,
    ExcelLoader,
    MarkdownLoader,
    PdfLoader,
    PowerPointLoader,
    TextLoader,
)
from offline_rag.parsers import (
    CODE_EXTENSIONS,
    CodeParser,
    DocxParser,
    HtmlParser,
    MarkdownParser,
    OcrPdfParser,
    PdfParser,
    PptxParser,
    StructuredTextParser,
    TextParser,
    XlsxParser,
)
from offline_rag.ports import EmbeddingProvider, SparseEmbeddingProvider, VectorStorePort
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
    ".xlsx",
    ".xlsm",
    ".pptx",
    ".pptm",
) + CODE_EXTENSIONS


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


@dataclass(frozen=True, slots=True)
class _ConfiguredChunker:
    callback: Callable[[ParsedDocument], Sequence[ChunkDraft]]

    def split(self, document: ParsedDocument) -> Sequence[ChunkDraft]:
        return self.callback(document)


class IngestionService:
    def __init__(
        self,
        config: RagConfig,
        *,
        embedding: EmbeddingProvider | None = None,
        sparse_embedding: SparseEmbeddingProvider | None = None,
        vector_store: VectorStorePort | None = None,
        token_counter: Callable[[str], int] | None = None,
        journal: IngestionJournal | None = None,
    ) -> None:
        self._config = config
        self._embedding = embedding
        self._sparse_embedding = sparse_embedding
        self._vector_store = vector_store
        self._token_counter = token_counter
        self._journal = journal
        self._text_loader = TextLoader()
        self._markdown_loader = MarkdownLoader()
        self._pdf_loader = PdfLoader()
        self._docx_loader = DocxLoader()
        self._excel_loader = ExcelLoader()
        self._powerpoint_loader = PowerPointLoader()
        self._structured_text_loader = TextLoader(
            extensions=(".html", ".htm", ".csv", ".json", ".jsonl"),
            media_types=("text/html", "text/csv", "application/json", "application/x-ndjson"),
        )
        self._code_loader = TextLoader(extensions=CODE_EXTENSIONS)
        self._text_parser = TextParser()
        self._markdown_parser = MarkdownParser()
        self._pdf_parser = PdfParser()
        self._ocr_pdf_parser = OcrPdfParser()
        self._docx_parser = DocxParser()
        self._xlsx_parser = XlsxParser()
        self._pptx_parser = PptxParser()
        self._html_parser = HtmlParser()
        self._code_parser = CodeParser()
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
        index_version = specification.fingerprint()[:16]
        job_id = self._start_job(index_version, prefix="build")
        results: list[IngestionItemResult] = []
        try:
            self._vector_store.ensure_index(specification)
            preview = self.preview(inputs, root=root)
            for failure in preview.failures:
                results.append(
                    IngestionItemResult(
                        source_uri=failure.source_uri,
                        stage=IngestionStage.FAILED,
                        failure=failure,
                    )
                )
                self._record_failure(job_id, None, failure)
            for item in preview.items:
                results.append(self._index_item(job_id, item, index_version=index_version))
        except Exception:
            self._finish_job(job_id, "failed")
            raise
        self._finish_job(
            job_id,
            "completed_with_errors"
            if any(result.stage is IngestionStage.FAILED for result in results)
            else "completed",
        )
        return IngestionReport(
            job_id=job_id,
            index_version=index_version,
            items=results,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def sync(
        self,
        inputs: Sequence[str | Path],
        specification: IndexSpecification,
        *,
        root: str | Path | None = None,
    ) -> IngestionReport:
        if self._embedding is None or self._vector_store is None or self._journal is None:
            raise RuntimeError("sync requires embedding, vector_store and journal components")
        started = time.perf_counter()
        index_version = specification.fingerprint()[:16]
        job_id = self._start_job(index_version, prefix="sync")
        results: list[IngestionItemResult] = []
        try:
            self._vector_store.ensure_index(specification)
            discovered = tuple(self._sources(inputs, root=root))
            plan = plan_sync(discovered, self._journal.list_sources())
            for source, _previous in plan.unchanged:
                results.append(
                    IngestionItemResult(source_uri=source.uri, stage=IngestionStage.SKIPPED)
                )
                self._journal.record_event(
                    job_id,
                    source_uri=source.uri,
                    source_id=source.source_id,
                    stage=IngestionStage.SKIPPED,
                )
            for source in plan.new:
                results.append(self._prepare_and_index(job_id, source, index_version))
            for source, previous in plan.modified:
                result = self._prepare_and_index(job_id, source, index_version, previous=previous)
                results.append(result)
            if self._config.ingestion.delete_missing_on_sync:
                for previous in plan.missing:
                    self._vector_store.delete(previous.chunk_ids)
                    self._journal.remove_source(previous.source_id)
                    self._journal.record_event(
                        job_id,
                        source_uri=previous.source_uri,
                        source_id=previous.source_id,
                        stage=IngestionStage.DELETED,
                    )
                    results.append(
                        IngestionItemResult(
                            source_uri=previous.source_uri,
                            stage=IngestionStage.DELETED,
                        )
                    )
        except Exception:
            self._finish_job(job_id, "failed")
            raise
        self._finish_job(
            job_id,
            "completed_with_errors"
            if any(result.stage is IngestionStage.FAILED for result in results)
            else "completed",
        )
        return IngestionReport(
            job_id=job_id,
            index_version=index_version,
            items=results,
            duration_ms=(time.perf_counter() - started) * 1000,
        )

    def _prepare_and_index(
        self,
        job_id: str,
        source: SourceDescriptor,
        index_version: str,
        *,
        previous: IndexedSource | None = None,
    ) -> IngestionItemResult:
        started = time.perf_counter()
        try:
            document = self._parse(source)
            item = PreviewItem(source, document, self._chunk(document))
            result = self._index_item(
                job_id,
                item,
                index_version=index_version,
                record_journal=previous is None,
            )
            if result.stage is IngestionStage.FAILED:
                return result
            if previous is not None and self._vector_store is not None:
                current_ids = {chunk.chunk_id for chunk in item.chunks}
                stale_ids = tuple(
                    chunk_id for chunk_id in previous.chunk_ids if chunk_id not in current_ids
                )
                self._vector_store.delete(stale_ids)
                if self._journal is not None:
                    self._journal.record_indexed(
                        job_id,
                        item.source,
                        document_id=item.document.document_id,
                        chunk_ids=tuple(chunk.chunk_id for chunk in item.chunks),
                        index_version=index_version,
                    )
            return result
        except Exception as exc:
            failure = self._failure(source.uri, IngestionStage.EMBEDDED, exc)
            self._record_failure(job_id, source.source_id, failure)
            if not self._config.ingestion.continue_on_error:
                raise
            return IngestionItemResult(
                source_uri=source.uri,
                stage=IngestionStage.FAILED,
                duration_ms=(time.perf_counter() - started) * 1000,
                failure=failure,
            )

    def _index_item(
        self,
        job_id: str,
        item: PreviewItem,
        *,
        index_version: str,
        record_journal: bool = True,
    ) -> IngestionItemResult:
        if self._embedding is None or self._vector_store is None:
            raise RuntimeError("indexing components are not configured")
        item_started = time.perf_counter()
        try:
            vectors = self._embedding.embed_documents(
                [chunk.embedding_text for chunk in item.chunks]
            )
            if len(vectors) != len(item.chunks):
                raise RuntimeError("embedding batch size does not match chunk count")
            sparse_vectors = (
                self._sparse_embedding.embed_documents(
                    [chunk.embedding_text for chunk in item.chunks]
                )
                if self._sparse_embedding is not None
                else tuple(((), ()) for _ in item.chunks)
            )
            if len(sparse_vectors) != len(item.chunks):
                raise RuntimeError("sparse embedding batch size does not match chunk count")
            records = [
                VectorRecord(
                    chunk.chunk_id,
                    vector,
                    chunk_to_payload(chunk),
                    sparse_indices=sparse[0],
                    sparse_values=sparse[1],
                )
                for chunk, vector, sparse in zip(item.chunks, vectors, sparse_vectors, strict=True)
            ]
            batch_size = self._config.ingestion.commit_batch_size
            for offset in range(0, len(records), batch_size):
                self._vector_store.upsert(records[offset : offset + batch_size])
            if self._journal is not None and record_journal:
                self._journal.record_indexed(
                    job_id,
                    item.source,
                    document_id=item.document.document_id,
                    chunk_ids=tuple(chunk.chunk_id for chunk in item.chunks),
                    index_version=index_version,
                )
            return IngestionItemResult(
                source_uri=item.source.uri,
                stage=IngestionStage.INDEXED,
                chunk_count=len(item.chunks),
                duration_ms=(time.perf_counter() - item_started) * 1000,
            )
        except Exception as exc:
            failure = self._failure(item.source.uri, IngestionStage.EMBEDDED, exc)
            self._record_failure(job_id, item.source.source_id, failure)
            if not self._config.ingestion.continue_on_error:
                raise
            return IngestionItemResult(
                source_uri=item.source.uri,
                stage=IngestionStage.FAILED,
                duration_ms=(time.perf_counter() - item_started) * 1000,
                failure=failure,
            )

    def _start_job(self, index_version: str, *, prefix: str) -> str:
        if self._journal is not None:
            return self._journal.start_job(index_version)
        return f"{prefix}-{time.time_ns()}"

    def _finish_job(self, job_id: str, status: str) -> None:
        if self._journal is not None:
            self._journal.finish_job(job_id, status=status)

    def _record_failure(self, job_id: str, source_id: str | None, failure: ItemFailure) -> None:
        if self._journal is not None:
            self._journal.record_event(
                job_id,
                source_uri=failure.source_uri,
                source_id=source_id,
                stage=IngestionStage.FAILED,
                error_code=failure.error_code,
                message=failure.message,
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
            loaded = self._pdf_loader.load(source)
            try:
                return self._pdf_parser.parse(loaded)
            except Exception as exc:
                if isinstance(exc, OfflineRagError) and "OCR" in str(exc):
                    return self._ocr_pdf_parser.parse(loaded)
                raise
        if suffix == ".docx":
            return self._docx_parser.parse(self._docx_loader.load(source))
        if suffix in (".xlsx", ".xlsm"):
            return self._xlsx_parser.parse(self._excel_loader.load(source))
        if suffix in (".pptx", ".pptm"):
            return self._pptx_parser.parse(self._powerpoint_loader.load(source))
        if suffix in (".html", ".htm"):
            return self._html_parser.parse(self._structured_text_loader.load(source))
        if suffix in (".csv", ".json", ".jsonl"):
            return StructuredTextParser(suffix.removeprefix(".")).parse(
                self._structured_text_loader.load(source)
            )
        if suffix in CODE_EXTENSIONS:
            return self._code_parser.parse(self._code_loader.load(source))
        raise UnsupportedDocumentError(
            "no built-in parser supports this source",
            details={"source_id": source.source_id, "extension": suffix},
        )

    def _chunk(self, document: ParsedDocument) -> tuple[Chunk, ...]:
        name = self._profile_for(document.source)
        drafts = self._split_with_profile(document, name, ())
        profile = self._config.chunk_profiles[name]
        if getattr(profile, "include_heading_in_embedding", True):
            drafts = HeadingContextInjector().process(drafts)
        drafts = ChunkDeduplicator().process(drafts)
        return self._finalizer.finalize(drafts)

    def _split_with_profile(
        self,
        document: ParsedDocument,
        name: str,
        resolving: tuple[str, ...],
    ) -> Sequence[ChunkDraft]:
        if name in resolving:
            chain = " -> ".join((*resolving, name))
            raise ConfigurationError(f"cyclic chunk profile fallback: {chain}")
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
        elif isinstance(profile, SyntaxChunkProfile):
            fallback = _ConfiguredChunker(
                lambda value: self._split_with_profile(
                    value, profile.fallback_profile, (*resolving, name)
                )
            )
            drafts = SyntaxChunker(fallback, max_chunk_size=profile.max_chunk_size).split(document)
        elif isinstance(profile, SemanticChunkProfile):
            if self._embedding is None:
                raise ConfigurationError(
                    "semantic chunking requires the configured local embedding provider"
                )
            drafts = SemanticChunker(
                self._embedding.embed_documents,
                similarity_threshold=profile.similarity_threshold,
                minimum_chunk_size=profile.minimum_chunk_size,
                maximum_chunk_size=profile.maximum_chunk_size,
            ).split(document)
        else:
            raise UnsupportedDocumentError(f"chunk profile type is not implemented: {profile.type}")
        return drafts

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
        if extension in (".csv", ".xlsx", ".xlsm") and "table_rows" in self._config.chunk_profiles:
            return "table_rows"
        if extension in (".pdf", ".pptx") and "page_aware" in self._config.chunk_profiles:
            return "page_aware"
        if extension in CODE_EXTENSIONS and "source_code" in self._config.chunk_profiles:
            return "source_code"
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
    config: RagConfig,
    embedding: EmbeddingSpecification,
    sparse_embedding: SparseEmbeddingProvider | None = None,
) -> IndexSpecification:
    dumped = config.model_dump(mode="json")
    chunking = cast(
        Mapping[str, JSONValue],
        {"profiles": dumped["chunk_profiles"], "routing": dumped["routing"]},
    )
    return IndexSpecification(
        index_format_version=1,
        payload_schema_version=1,
        embedding=embedding,
        parser_versions={
            "text": TextParser.version,
            "markdown": MarkdownParser.version,
            "pdf": PdfParser.version,
            "pdf_ocr": OcrPdfParser.version,
            "docx": DocxParser.version,
            "xlsx": XlsxParser.version,
            "pptx": PptxParser.version,
            "html": HtmlParser.version,
            "structured_text": StructuredTextParser.version,
            "code": CodeParser.version,
        },
        chunking_configuration=chunking,
        sparse_embedding=(sparse_embedding.specification if sparse_embedding is not None else None),
    )
